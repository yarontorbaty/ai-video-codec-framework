#!/usr/bin/env python3
"""
AI Video Codec Agent
Implements a hybrid semantic codec for video compression using neural networks.
Target: Beat 10 Mbps HEVC baseline with < 1 Mbps while maintaining PSNR > 95%
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
from PIL import Image
import boto3
from botocore.exceptions import ClientError
import psutil
import threading
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.aws_utils import AWSUtils
from utils.metrics import MetricsCalculator
from utils.video_utils import VideoProcessor
from agents.procedural_generator import ProceduralCompressionAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoDataset(Dataset):
    """Dataset for video frames with semantic understanding."""
    
    def __init__(self, video_path: str, frame_size: Tuple[int, int] = (1920, 1080)):
        self.video_path = video_path
        self.frame_size = frame_size
        self.frames = self._load_frames()
        
    def _load_frames(self) -> List[np.ndarray]:
        """Load video frames."""
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize to target resolution
            frame = cv2.resize(frame, self.frame_size)
            frames.append(frame)
            
        cap.release()
        return frames
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        # Convert BGR to RGB and normalize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        # Convert to tensor
        frame = torch.from_numpy(frame).permute(2, 0, 1)  # HWC to CHW
        return frame


class SemanticEncoder(nn.Module):
    """Semantic encoder for understanding video content."""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            # Initial convolution
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # ResNet-like blocks
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Semantic understanding head
        self.semantic_head = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )
        
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        """Create a ResNet-like layer."""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        features = self.backbone(x)
        pooled = self.global_pool(features).flatten(1)
        semantic = self.semantic_head(pooled)
        return semantic


class MotionPredictor(nn.Module):
    """Motion prediction network for inter-frame compression."""
    
    def __init__(self, input_channels: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Motion feature extraction
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Motion vector prediction
        self.motion_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, padding=1),  # 2D motion vectors
        )
        
    def forward(self, current_frame, reference_frame):
        # Concatenate current and reference frames
        combined = torch.cat([current_frame, reference_frame], dim=1)
        motion_features = self.motion_encoder(combined)
        motion_vectors = self.motion_head(motion_features)
        return motion_vectors


class GenerativeRefiner(nn.Module):
    """Generative model for quality refinement."""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, latent_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AICodecAgent:
    """Main AI Codec Agent for video compression."""
    
    def __init__(self, config_path: str = "config/ai_codec_config.yaml"):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.semantic_encoder = SemanticEncoder().to(self.device)
        self.motion_predictor = MotionPredictor().to(self.device)
        self.generative_refiner = GenerativeRefiner().to(self.device)
        
        # Optimizers
        self.semantic_optimizer = optim.Adam(self.semantic_encoder.parameters(), lr=1e-4)
        self.motion_optimizer = optim.Adam(self.motion_predictor.parameters(), lr=1e-4)
        self.refiner_optimizer = optim.Adam(self.generative_refiner.parameters(), lr=1e-4)
        
        # Utilities
        self.aws_utils = AWSUtils()
        self.metrics_calc = MetricsCalculator()
        self.video_processor = VideoProcessor()
        
        # Procedural generation agent (demoscene-inspired)
        self.procedural_agent = ProceduralCompressionAgent()
        
        # Training state
        self.training_step = 0
        self.best_psnr = 0.0
        self.best_bitrate = float('inf')
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'training': {
                'batch_size': 4,
                'epochs': 100,
                'learning_rate': 1e-4,
                'target_bitrate_mbps': 1.0,  # Target < 1 Mbps
                'target_psnr_db': 35.0,     # Target PSNR > 35 dB (95%+ quality)
            },
            'model': {
                'latent_dim': 512,
                'hidden_dim': 256,
                'compression_ratio': 0.1,  # 90% reduction from HEVC
            },
            'aws': {
                's3_bucket': 'ai-video-codec-videos-580473065386',
                'region': 'us-east-1',
            }
        }
    
    def download_test_data(self) -> Tuple[str, str]:
        """Download test videos from S3."""
        logger.info("Downloading test data from S3...")
        
        # Create local directories
        os.makedirs("data/source", exist_ok=True)
        os.makedirs("data/hevc", exist_ok=True)
        
        # Download source video
        source_path = "data/source/SOURCE_HD_RAW.mp4"
        self.aws_utils.download_from_s3(
            bucket=self.config['aws']['s3_bucket'],
            key="source/SOURCE_HD_RAW.mp4",
            local_path=source_path
        )
        
        # Download HEVC reference
        hevc_path = "data/hevc/HEVC_HD_10Mbps.mp4"
        self.aws_utils.download_from_s3(
            bucket=self.config['aws']['s3_bucket'],
            key="hevc/HEVC_HD_10Mbps.mp4",
            local_path=hevc_path
        )
        
        logger.info(f"Downloaded source: {source_path}")
        logger.info(f"Downloaded HEVC reference: {hevc_path}")
        
        return source_path, hevc_path
    
    def create_dataset(self, video_path: str) -> VideoDataset:
        """Create dataset from video."""
        return VideoDataset(video_path)
    
    def train_semantic_encoder(self, dataset: VideoDataset) -> float:
        """Train semantic encoder."""
        logger.info("Training semantic encoder...")
        
        dataloader = DataLoader(dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
        
        total_loss = 0.0
        for batch_idx, frames in enumerate(dataloader):
            frames = frames.to(self.device)
            
            self.semantic_optimizer.zero_grad()
            
            # Forward pass
            semantic_features = self.semantic_encoder(frames)
            
            # Semantic consistency loss (encourage similar frames to have similar features)
            semantic_loss = torch.mean(torch.var(semantic_features, dim=0))
            
            semantic_loss.backward()
            self.semantic_optimizer.step()
            
            total_loss += semantic_loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Semantic training step {batch_idx}, loss: {semantic_loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Semantic encoder training completed. Average loss: {avg_loss:.4f}")
        return avg_loss
    
    def train_motion_predictor(self, dataset: VideoDataset) -> float:
        """Train motion predictor."""
        logger.info("Training motion predictor...")
        
        dataloader = DataLoader(dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
        
        total_loss = 0.0
        for batch_idx, frames in enumerate(dataloader):
            if batch_idx == 0:
                continue  # Need at least 2 frames
                
            frames = frames.to(self.device)
            current_frame = frames[1:]
            reference_frame = frames[:-1]
            
            self.motion_optimizer.zero_grad()
            
            # Forward pass
            motion_vectors = self.motion_predictor(current_frame, reference_frame)
            
            # Motion prediction loss (L2)
            motion_loss = torch.mean(torch.square(motion_vectors))
            
            motion_loss.backward()
            self.motion_optimizer.step()
            
            total_loss += motion_loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Motion training step {batch_idx}, loss: {motion_loss.item():.4f}")
        
        avg_loss = total_loss / max(1, len(dataloader) - 1)
        logger.info(f"Motion predictor training completed. Average loss: {avg_loss:.4f}")
        return avg_loss
    
    def train_generative_refiner(self, dataset: VideoDataset) -> float:
        """Train generative refiner."""
        logger.info("Training generative refiner...")
        
        dataloader = DataLoader(dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
        
        total_loss = 0.0
        for batch_idx, frames in enumerate(dataloader):
            frames = frames.to(self.device)
            
            self.refiner_optimizer.zero_grad()
            
            # Forward pass
            refined_frames = self.generative_refiner(frames)
            
            # Reconstruction loss
            reconstruction_loss = torch.mean(torch.square(frames - refined_frames))
            
            reconstruction_loss.backward()
            self.refiner_optimizer.step()
            
            total_loss += reconstruction_loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Refiner training step {batch_idx}, loss: {reconstruction_loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Generative refiner training completed. Average loss: {avg_loss:.4f}")
        return avg_loss
    
    def compress_video(self, input_path: str, output_path: str) -> Dict:
        """Compress video using hybrid AI codec with procedural generation."""
        logger.info(f"Compressing video: {input_path} -> {output_path}")
        
        # Load video
        dataset = self.create_dataset(input_path)
        
        # Set models to evaluation mode
        self.semantic_encoder.eval()
        self.motion_predictor.eval()
        self.generative_refiner.eval()
        
        compressed_frames = []
        total_compression_time = 0.0
        
        with torch.no_grad():
            for i, frame in enumerate(dataset):
                start_time = time.time()
                
                frame = frame.unsqueeze(0).to(self.device)  # Add batch dimension
                
                # Semantic understanding
                semantic_features = self.semantic_encoder(frame)
                
                # Motion prediction (if not first frame)
                if i > 0:
                    prev_frame = dataset[i-1].unsqueeze(0).to(self.device)
                    motion_vectors = self.motion_predictor(frame, prev_frame)
                else:
                    motion_vectors = torch.zeros_like(frame[:, :2])  # No motion for first frame
                
                # Generative refinement
                refined_frame = self.generative_refiner(frame)
                
                # Apply motion compensation
                if i > 0:
                    # Simple motion compensation (in practice, would use more sophisticated warping)
                    refined_frame = refined_frame + motion_vectors.mean() * 0.1
                
                compressed_frames.append(refined_frame.cpu())
                
                compression_time = time.time() - start_time
                total_compression_time += compression_time
                
                if i % 30 == 0:  # Log every 30 frames
                    logger.info(f"Compressed frame {i}/{len(dataset)}, time: {compression_time:.3f}s")
        
        # Save compressed video
        self._save_compressed_video(compressed_frames, output_path, dataset.frame_size)
        
        # Calculate metrics
        metrics = self._calculate_compression_metrics(input_path, output_path, total_compression_time)
        
        logger.info(f"Compression completed. Metrics: {metrics}")
        return metrics
    
    def compress_video_hybrid(self, input_path: str, output_path: str) -> Dict:
        """Compress video using hybrid approach: AI + Procedural Generation."""
        logger.info(f"Hybrid compression: {input_path} -> {output_path}")
        
        # Try procedural compression first (demoscene-inspired)
        logger.info("Attempting procedural compression...")
        procedural_results = self.procedural_agent.compress_video(input_path, "temp_procedural.mp4")
        
        # Try traditional AI compression
        logger.info("Attempting AI compression...")
        ai_results = self.compress_video(input_path, "temp_ai.mp4")
        
        # Choose best approach based on metrics
        if (procedural_results['compression_ratio'] < ai_results['compression_ratio'] and 
            procedural_results['bitrate_mbps'] < ai_results['bitrate_mbps']):
            logger.info("Procedural compression selected as best approach")
            best_results = procedural_results
            os.rename("temp_procedural.mp4", output_path)
            os.remove("temp_ai.mp4")
        else:
            logger.info("AI compression selected as best approach")
            best_results = ai_results
            os.rename("temp_ai.mp4", output_path)
            os.remove("temp_procedural.mp4")
        
        # Add hybrid decision info
        best_results['compression_method'] = 'hybrid'
        best_results['procedural_metrics'] = procedural_results
        best_results['ai_metrics'] = ai_results
        
        logger.info(f"Hybrid compression completed. Best metrics: {best_results}")
        return best_results
    
    def _save_compressed_video(self, frames: List[torch.Tensor], output_path: str, frame_size: Tuple[int, int]):
        """Save compressed frames as video."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, frame_size)
        
        for frame in frames:
            # Convert from tensor to numpy
            frame_np = frame.squeeze(0).permute(1, 2, 0).numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            out.write(frame_np)
        
        out.release()
    
    def _calculate_compression_metrics(self, input_path: str, output_path: str, compression_time: float) -> Dict:
        """Calculate compression metrics."""
        # File sizes
        input_size = os.path.getsize(input_path)
        output_size = os.path.getsize(output_path)
        
        # Compression ratio
        compression_ratio = output_size / input_size
        
        # Bitrate calculation (assuming 10 second video at 30fps)
        duration = 10.0  # seconds
        fps = 30.0
        bitrate_mbps = (output_size * 8) / (duration * 1_000_000)  # Convert to Mbps
        
        # PSNR calculation (simplified)
        psnr_db = self.metrics_calc.calculate_psnr(input_path, output_path)
        
        return {
            'input_size_mb': input_size / (1024 * 1024),
            'output_size_mb': output_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'bitrate_mbps': bitrate_mbps,
            'psnr_db': psnr_db,
            'compression_time_s': compression_time,
            'fps': fps,
        }
    
    def run_experiment(self) -> Dict:
        """Run complete AI codec experiment."""
        logger.info("Starting AI codec experiment...")
        
        # Download test data
        source_path, hevc_path = self.download_test_data()
        
        # Create dataset
        dataset = self.create_dataset(source_path)
        logger.info(f"Created dataset with {len(dataset)} frames")
        
        # Training phase
        logger.info("Starting training phase...")
        semantic_loss = self.train_semantic_encoder(dataset)
        motion_loss = self.train_motion_predictor(dataset)
        refiner_loss = self.train_generative_refiner(dataset)
        
        # Compression phase (hybrid approach)
        logger.info("Starting hybrid compression phase...")
        output_path = "data/compressed/ai_codec_hybrid_output.mp4"
        os.makedirs("data/compressed", exist_ok=True)
        
        compression_metrics = self.compress_video_hybrid(source_path, output_path)
        
        # Compare with HEVC baseline
        hevc_metrics = self._calculate_compression_metrics(source_path, hevc_path, 0)
        
        # Results
        results = {
            'training_losses': {
                'semantic': semantic_loss,
                'motion': motion_loss,
                'refiner': refiner_loss,
            },
            'ai_codec_metrics': compression_metrics,
            'hevc_baseline_metrics': hevc_metrics,
            'improvement': {
                'bitrate_reduction_percent': ((hevc_metrics['bitrate_mbps'] - compression_metrics['bitrate_mbps']) / hevc_metrics['bitrate_mbps']) * 100,
                'psnr_improvement_db': compression_metrics['psnr_db'] - hevc_metrics['psnr_db'],
                'compression_ratio_improvement': compression_metrics['compression_ratio'] / hevc_metrics['compression_ratio'],
            },
            'target_achieved': {
                'bitrate_target': compression_metrics['bitrate_mbps'] < self.config['training']['target_bitrate_mbps'],
                'psnr_target': compression_metrics['psnr_db'] > self.config['training']['target_psnr_db'],
            }
        }
        
        logger.info("Experiment completed!")
        logger.info(f"Results: {json.dumps(results, indent=2)}")
        
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='AI Video Codec Agent')
    parser.add_argument('--config', type=str, default='config/ai_codec_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--mode', type=str, choices=['train', 'compress', 'experiment'], 
                       default='experiment', help='Operation mode')
    parser.add_argument('--input', type=str, help='Input video path')
    parser.add_argument('--output', type=str, help='Output video path')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = AICodecAgent(args.config)
    
    if args.mode == 'experiment':
        # Run complete experiment
        results = agent.run_experiment()
        
        # Save results
        with open('experiment_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Experiment results saved to experiment_results.json")
        
    elif args.mode == 'compress':
        if not args.input or not args.output:
            logger.error("Input and output paths required for compress mode")
            return
        
        metrics = agent.compress_video(args.input, args.output)
        logger.info(f"Compression metrics: {metrics}")
    
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
