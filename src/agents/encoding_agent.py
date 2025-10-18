#!/usr/bin/env python3
"""
Encoding Agent - GPU-Accelerated Video Compression
Responsible for analyzing scenes and compressing video using adaptive strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SceneClassifier(nn.Module):
    """Classifies scene type for adaptive compression strategy selection."""
    
    def __init__(self, num_classes=5):
        super().__init__()
        # Lightweight CNN for scene classification
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier heads
        self.fc_scene_type = nn.Linear(256 * 4 * 4, num_classes)
        self.fc_complexity = nn.Linear(256 * 4 * 4, 1)
        self.fc_motion = nn.Linear(256 * 4 * 4, 1)
        
    def forward(self, frames: torch.Tensor) -> Dict:
        """
        Analyze scene characteristics.
        
        Args:
            frames: [B, T, C, H, W] video frames
            
        Returns:
            Scene analysis dictionary
        """
        # Use middle frame for scene classification
        B, T, C, H, W = frames.shape
        middle_frame = frames[:, T//2, :, :, :]  # [B, C, H, W]
        
        # Extract features
        x = F.relu(self.conv1(middle_frame))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(B, -1)
        
        # Classify
        scene_logits = self.fc_scene_type(x)
        complexity = torch.sigmoid(self.fc_complexity(x))
        motion = torch.sigmoid(self.fc_motion(x))
        
        # Scene types: 0=static, 1=talking_head, 2=moderate_motion, 3=high_motion, 4=synthetic
        scene_type_idx = torch.argmax(scene_logits, dim=1)
        scene_types = ['static', 'talking_head', 'moderate_motion', 'high_motion', 'synthetic']
        
        return {
            'scene_type': scene_types[scene_type_idx[0].item()],
            'scene_type_idx': scene_type_idx,
            'complexity': complexity[0].item(),
            'motion_intensity': motion[0].item(),
            'scene_logits': scene_logits
        }


class IFrameVAE(nn.Module):
    """Variational Autoencoder for I-frame compression."""
    
    def __init__(self, latent_dim=512, input_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: 1080p → latent
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),  # 1920x1080 → 960x540
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1),  # 960x540 → 480x270
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),  # 480x270 → 240x135
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1),  # 240x135 → 120x67
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, 4, 2, 1),  # 120x67 → 60x33
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Calculate flattened size (60x33x512)
        self.flat_size = 60 * 33 * 512
        
        # Latent space
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode frame to latent distribution.
        
        Args:
            x: [B, C, H, W] input frame
            
        Returns:
            mu, logvar: Latent distribution parameters
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Returns:
            latent, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class SemanticDescriptionGenerator(nn.Module):
    """Generates semantic descriptions for video content."""
    
    def __init__(self, latent_dim=512, description_dim=256):
        super().__init__()
        
        # Visual encoder (similar to scene classifier but deeper)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 4, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Temporal encoder for motion
        self.temporal_encoder = nn.LSTM(
            input_size=512 * 8 * 8,
            hidden_size=512,
            num_layers=2,
            batch_first=True
        )
        
        # Semantic projection
        self.semantic_projection = nn.Linear(512, description_dim)
        
        # Motion vector predictor
        self.motion_predictor = nn.Linear(512, 2)  # dx, dy per frame
        
    def forward(self, frames: torch.Tensor, i_frame_indices: List[int]) -> Dict:
        """
        Generate semantic description for video segment.
        
        Args:
            frames: [B, T, C, H, W] video frames
            i_frame_indices: List of I-frame indices
            
        Returns:
            Semantic description dictionary
        """
        B, T, C, H, W = frames.shape
        
        # Encode all frames visually
        frames_flat = frames.view(B * T, C, H, W)
        visual_features = self.visual_encoder(frames_flat)  # [B*T, 512, 8, 8]
        visual_features = visual_features.view(B, T, -1)  # [B, T, 512*8*8]
        
        # Encode temporal dynamics
        temporal_features, _ = self.temporal_encoder(visual_features)  # [B, T, 512]
        
        # Generate semantic embedding (use mean over time)
        semantic_embedding = self.semantic_projection(
            temporal_features.mean(dim=1)
        )  # [B, description_dim]
        
        # Predict motion vectors for each frame
        motion_vectors = self.motion_predictor(temporal_features)  # [B, T, 2]
        
        return {
            'semantic_embedding': semantic_embedding,
            'temporal_features': temporal_features,
            'motion_vectors': motion_vectors,
            'visual_features': visual_features
        }


class CompressionStrategySelector:
    """Selects optimal compression strategy based on scene analysis."""
    
    STRATEGIES = {
        'semantic_latent': {'bitrate_estimate': 0.3, 'quality_estimate': 0.90},
        'i_frame_interpolation': {'bitrate_estimate': 0.5, 'quality_estimate': 0.92},
        'hybrid_semantic': {'bitrate_estimate': 0.8, 'quality_estimate': 0.95},
        'av1': {'bitrate_estimate': 2.0, 'quality_estimate': 0.98},
        'x265': {'bitrate_estimate': 2.5, 'quality_estimate': 0.98}
    }
    
    def select_strategy(self, scene_info: Dict, config: Dict) -> str:
        """
        Select optimal compression strategy.
        
        Args:
            scene_info: Scene analysis from SceneClassifier
            config: Compression configuration
            
        Returns:
            Strategy name
        """
        scene_type = scene_info['scene_type']
        complexity = scene_info['complexity']
        motion = scene_info['motion_intensity']
        
        target_bitrate = config.get('target_bitrate_mbps', 1.0)
        target_quality = config.get('target_quality_ssim', 0.95)
        
        logger.info(f"  Scene: {scene_type}, Complexity: {complexity:.2f}, Motion: {motion:.2f}")
        
        # Decision logic
        
        # Static or very low motion → Semantic latent (ultra-low bitrate)
        if motion < 0.15:
            logger.info(f"  → Strategy: semantic_latent (low motion)")
            return 'semantic_latent'
        
        # Talking head with moderate motion → I-frame interpolation
        if scene_type == 'talking_head' and motion < 0.4:
            logger.info(f"  → Strategy: i_frame_interpolation (talking head)")
            return 'i_frame_interpolation'
        
        # High motion → Need traditional codec for quality
        if motion > 0.7:
            if target_bitrate > 1.5:
                logger.info(f"  → Strategy: av1 (high motion, budget allows)")
                return 'av1'
            else:
                logger.info(f"  → Strategy: hybrid_semantic (high motion, tight budget)")
                return 'hybrid_semantic'
        
        # Synthetic content → Can use procedural approaches
        if scene_type == 'synthetic':
            logger.info(f"  → Strategy: semantic_latent (synthetic)")
            return 'semantic_latent'
        
        # Default: Hybrid approach balancing bitrate and quality
        logger.info(f"  → Strategy: hybrid_semantic (default)")
        return 'hybrid_semantic'


class EncodingAgent(nn.Module):
    """
    Main encoding agent that coordinates scene analysis, I-frame compression,
    and semantic description generation.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Sub-modules
        self.scene_classifier = SceneClassifier()
        self.i_frame_encoder = IFrameVAE(latent_dim=config.get('latent_dim', 512))
        self.semantic_generator = SemanticDescriptionGenerator(
            latent_dim=config.get('latent_dim', 512),
            description_dim=config.get('description_dim', 256)
        )
        self.strategy_selector = CompressionStrategySelector()
        
        # I-frame selection parameters
        self.i_frame_interval = config.get('i_frame_interval', 30)  # Every 30 frames
        
    def select_i_frames(self, frames: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        """
        Select I-frames intelligently based on scene changes.
        
        Args:
            frames: [B, T, C, H, W] video frames
            
        Returns:
            i_frame_indices, i_frames tensor
        """
        B, T, C, H, W = frames.shape
        
        # For now, use fixed interval (can be improved with scene change detection)
        i_frame_indices = list(range(0, T, self.i_frame_interval))
        if i_frame_indices[-1] != T - 1:
            i_frame_indices.append(T - 1)  # Always include last frame
        
        i_frames = frames[:, i_frame_indices, :, :, :]  # [B, num_i_frames, C, H, W]
        
        logger.info(f"  Selected {len(i_frame_indices)} I-frames from {T} total frames")
        
        return i_frame_indices, i_frames
    
    def compress_i_frames(self, i_frames: torch.Tensor) -> Dict:
        """
        Compress I-frames using VAE.
        
        Args:
            i_frames: [B, N, C, H, W] I-frames
            
        Returns:
            Compressed I-frame data
        """
        B, N, C, H, W = i_frames.shape
        i_frames_flat = i_frames.view(B * N, C, H, W)
        
        # Encode to latent space
        latents, mu, logvar = self.i_frame_encoder(i_frames_flat)
        
        # Reshape back
        latents = latents.view(B, N, -1)
        mu = mu.view(B, N, -1)
        logvar = logvar.view(B, N, -1)
        
        return {
            'latents': latents,
            'mu': mu,
            'logvar': logvar
        }
    
    def forward(self, frames: torch.Tensor) -> Dict:
        """
        Main encoding forward pass.
        
        Args:
            frames: [B, T, C, H, W] video frames (normalized to [0, 1])
            
        Returns:
            Compressed video representation
        """
        B, T, C, H, W = frames.shape
        
        logger.info(f"🎬 Encoding video: {B}x{T} frames at {H}x{W}")
        
        # Step 1: Scene analysis
        logger.info(f"  📊 Analyzing scene...")
        scene_info = self.scene_classifier(frames)
        
        # Step 2: Select compression strategy
        strategy = self.strategy_selector.select_strategy(scene_info, self.config)
        
        # Step 3: Select and compress I-frames
        logger.info(f"  🎞️  Selecting I-frames...")
        i_frame_indices, i_frames = self.select_i_frames(frames)
        
        logger.info(f"  🗜️  Compressing I-frames...")
        i_frame_compressed = self.compress_i_frames(i_frames)
        
        # Step 4: Generate semantic description
        logger.info(f"  📝 Generating semantic description...")
        semantic_desc = self.semantic_generator(frames, i_frame_indices)
        
        # Step 5: Package compressed data
        compressed_data = {
            'strategy': strategy,
            'scene_info': scene_info,
            'i_frame_indices': i_frame_indices,
            'i_frame_compressed': i_frame_compressed,
            'semantic_description': semantic_desc,
            'metadata': {
                'num_frames': T,
                'resolution': (H, W),
                'i_frame_interval': self.i_frame_interval,
                'num_i_frames': len(i_frame_indices)
            }
        }
        
        logger.info(f"  ✅ Encoding complete")
        
        return compressed_data


def estimate_compressed_size(compressed_data: Dict) -> Tuple[int, float]:
    """
    Estimate compressed size in bytes.
    
    Args:
        compressed_data: Output from EncodingAgent
        
    Returns:
        size_bytes, size_mb
    """
    # I-frames: latent_dim * num_i_frames * 4 bytes (float32)
    latents = compressed_data['i_frame_compressed']['latents']
    i_frame_size = latents.numel() * 4
    
    # Semantic description: description_dim * 4 bytes
    semantic = compressed_data['semantic_description']['semantic_embedding']
    semantic_size = semantic.numel() * 4
    
    # Motion vectors: 2 * num_frames * 4 bytes
    motion = compressed_data['semantic_description']['motion_vectors']
    motion_size = motion.numel() * 4
    
    total_size = i_frame_size + semantic_size + motion_size
    size_mb = total_size / (1024 * 1024)
    
    logger.info(f"  📦 Compressed size estimate:")
    logger.info(f"     I-frames: {i_frame_size / 1024:.1f} KB")
    logger.info(f"     Semantic: {semantic_size / 1024:.1f} KB")
    logger.info(f"     Motion: {motion_size / 1024:.1f} KB")
    logger.info(f"     Total: {size_mb:.2f} MB")
    
    return total_size, size_mb


def calculate_bitrate(size_bytes: int, duration_seconds: float) -> float:
    """
    Calculate bitrate from compressed size.
    
    Args:
        size_bytes: Compressed size in bytes
        duration_seconds: Video duration in seconds
        
    Returns:
        Bitrate in Mbps
    """
    bitrate_mbps = (size_bytes * 8) / (duration_seconds * 1_000_000)
    return bitrate_mbps


# ============================================================
# MAIN COMPRESSION ENTRY POINT
# ============================================================

def compress_video_tensor(
    frames: torch.Tensor,
    config: Dict,
    device: str = 'cuda'
) -> Dict:
    """
    Compress video frames using EncodingAgent.
    
    Args:
        frames: [B, T, C, H, W] video frames (normalized to [0, 1])
        config: Configuration dict
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        {
            'compressed_data': compressed representation,
            'bitrate_mbps': estimated bitrate,
            'compression_ratio': estimated compression ratio,
            'strategy': compression strategy used
        }
    """
    # Move to device
    frames = frames.to(device)
    
    # Create encoding agent
    encoder = EncodingAgent(config).to(device)
    encoder.eval()
    
    # Compress
    with torch.no_grad():
        compressed_data = encoder(frames)
    
    # Estimate size and bitrate
    duration = config.get('duration', 10.0)
    fps = config.get('fps', 30.0)
    resolution = config.get('resolution', (1920, 1080))
    
    compressed_size_bytes, compressed_size_mb = estimate_compressed_size(compressed_data)
    bitrate_mbps = calculate_bitrate(compressed_size_bytes, duration)
    
    # Calculate original size (uncompressed RGB)
    num_frames = frames.shape[1]
    original_size_bytes = num_frames * resolution[0] * resolution[1] * 3
    compression_ratio = original_size_bytes / compressed_size_bytes
    
    logger.info(f"  🎯 Compression Results:")
    logger.info(f"     Bitrate: {bitrate_mbps:.4f} Mbps")
    logger.info(f"     Compression ratio: {compression_ratio:.1f}x")
    logger.info(f"     Strategy: {compressed_data['strategy']}")
    
    return {
        'compressed_data': compressed_data,
        'bitrate_mbps': bitrate_mbps,
        'compression_ratio': compression_ratio,
        'compressed_size_mb': compressed_size_mb,
        'strategy': compressed_data['strategy']
    }


