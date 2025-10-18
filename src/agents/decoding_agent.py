#!/usr/bin/env python3
"""
Decoding Agent - Lightweight Video Reconstruction
Designed to run on 40 TOPS edge devices (Qualcomm Snapdragon, Apple A17, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LightweightIFrameDecoder(nn.Module):
    """
    Lightweight decoder for I-frames from latent space.
    Optimized for 40 TOPS constraint using efficient architecture.
    """
    
    def __init__(self, latent_dim=512, output_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Project latent to spatial features
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        
        # Efficient decoder using depth-wise separable convolutions
        self.decoder = nn.Sequential(
            # 8x8 → 16x16
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 16x16 → 32x32
            self._depthwise_separable_transpose(512, 256, 4, 2, 1),
            
            # 32x32 → 64x64
            self._depthwise_separable_transpose(256, 128, 4, 2, 1),
            
            # 64x64 → 128x128
            self._depthwise_separable_transpose(128, 64, 4, 2, 1),
            
            # 128x128 → 256x256
            self._depthwise_separable_transpose(64, 32, 4, 2, 1),
            
            # 256x256 → 512x512
            self._depthwise_separable_transpose(32, 16, 4, 2, 1),
            
            # 512x512 → 1024x1024 (close to 1920x1080)
            self._depthwise_separable_transpose(16, output_channels, 4, 2, 1),
            
            nn.Sigmoid()  # Output [0, 1]
        )
        
    def _depthwise_separable_transpose(
        self, in_channels, out_channels, kernel_size, stride, padding
    ):
        """
        Depth-wise separable transposed convolution.
        Reduces computation by ~9x compared to standard conv.
        """
        return nn.Sequential(
            # Depth-wise
            nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size, stride, padding,
                groups=in_channels
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Point-wise
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to I-frame.
        
        Args:
            latent: [B, latent_dim] latent vectors
            
        Returns:
            I-frames: [B, C, H, W] reconstructed frames
        """
        # Project to spatial features
        x = self.fc(latent)
        x = x.view(-1, 512, 8, 8)
        
        # Decode to image
        x = self.decoder(x)
        
        # Resize to exact target resolution (1920x1080)
        x = F.interpolate(x, size=(1080, 1920), mode='bilinear', align_corners=False)
        
        return x


class LightweightVideoGenerator(nn.Module):
    """
    Generates intermediate frames between I-frames using semantic description.
    Optimized for 40 TOPS constraint.
    """
    
    def __init__(self, description_dim=256):
        super().__init__()
        
        # Semantic to visual features
        self.semantic_to_visual = nn.Sequential(
            nn.Linear(description_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512 * 4 * 4)
        )
        
        # Efficient frame generator using U-Net style architecture
        # Encoder
        self.enc1 = self._conv_block(3, 32)  # 1920x1080 → features
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        
        # Bottleneck with semantic conditioning
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128 + 512, 256, 3, 1, 1),  # +512 from semantic features
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with skip connections
        self.dec3 = self._conv_block(256, 64)  # 128 + 128 from skip
        self.dec2 = self._conv_block(128, 32)  # 64 + 64 from skip
        self.dec1 = self._conv_block(64, 16)   # 32 + 32 from skip
        
        # Final output
        self.final = nn.Conv2d(16, 3, 3, 1, 1)
        
    def _conv_block(self, in_channels, out_channels):
        """Efficient convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        ref_frame: torch.Tensor,
        semantic_desc: torch.Tensor,
        motion_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate frame from reference I-frame and semantic description.
        
        Args:
            ref_frame: [B, C, H, W] reference I-frame
            semantic_desc: [B, description_dim] semantic embedding
            motion_vector: [B, 2] motion vector (dx, dy)
            
        Returns:
            Generated frame [B, C, H, W]
        """
        B, C, H, W = ref_frame.shape
        
        # Prepare semantic features
        semantic_features = self.semantic_to_visual(semantic_desc)
        semantic_features = semantic_features.view(B, 512, 4, 4)
        semantic_features = F.interpolate(
            semantic_features,
            size=(H // 8, W // 8),
            mode='bilinear',
            align_corners=False
        )
        
        # Apply motion to reference frame (simple warp)
        warped_ref = self._warp_frame(ref_frame, motion_vector)
        
        # Encode reference frame
        e1 = self.enc1(warped_ref)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        
        # Bottleneck with semantic conditioning
        e3_downsampled = F.max_pool2d(e3, 2)
        bottleneck_input = torch.cat([e3_downsampled, semantic_features], dim=1)
        b = self.bottleneck(bottleneck_input)
        
        # Decode with skip connections
        d3 = self.dec3(torch.cat([F.interpolate(b, scale_factor=2, mode='nearest'), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode='nearest'), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode='nearest'), e1], dim=1))
        
        # Final output
        output = self.final(d1)
        output = torch.sigmoid(output)  # [0, 1] range
        
        return output
    
    def _warp_frame(self, frame: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        """
        Warp frame according to motion vector.
        
        Args:
            frame: [B, C, H, W]
            motion: [B, 2] motion vector (dx, dy) in pixels
            
        Returns:
            Warped frame
        """
        B, C, H, W = frame.shape
        
        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=frame.device),
            torch.linspace(-1, 1, W, device=frame.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Apply motion (normalized to [-1, 1])
        motion_normalized = motion.view(B, 1, 1, 2) / torch.tensor([W, H], device=frame.device) * 2
        grid_warped = grid + motion_normalized
        
        # Sample from warped grid
        warped = F.grid_sample(frame, grid_warped, align_corners=False, padding_mode='border')
        
        return warped


class TemporalConsistencyEnhancer(nn.Module):
    """
    Ensures temporal consistency between generated frames.
    Reduces flickering and artifacts.
    """
    
    def __init__(self, num_channels=3):
        super().__init__()
        
        # Temporal filter (lightweight 3D conv)
        self.temporal_filter = nn.Sequential(
            nn.Conv3d(num_channels, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, num_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Sigmoid()
        )
        
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Enhance temporal consistency.
        
        Args:
            frames: [B, T, C, H, W] video frames
            
        Returns:
            Enhanced frames [B, T, C, H, W]
        """
        # Reshape for 3D conv: [B, C, T, H, W]
        B, T, C, H, W = frames.shape
        frames_3d = frames.permute(0, 2, 1, 3, 4)
        
        # Apply temporal filter
        enhanced = self.temporal_filter(frames_3d)
        
        # Reshape back: [B, T, C, H, W]
        enhanced = enhanced.permute(0, 2, 1, 3, 4)
        
        # Blend with original (residual connection)
        output = 0.7 * frames + 0.3 * enhanced
        
        return output


class DecodingAgent(nn.Module):
    """
    Main decoding agent that reconstructs video from compressed representation.
    Optimized for 40 TOPS edge deployment.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Sub-modules
        self.i_frame_decoder = LightweightIFrameDecoder(
            latent_dim=config.get('latent_dim', 512),
            output_channels=3
        )
        self.video_generator = LightweightVideoGenerator(
            description_dim=config.get('description_dim', 256)
        )
        self.temporal_enhancer = TemporalConsistencyEnhancer(num_channels=3)
        
        # Whether to use temporal enhancement (adds compute cost)
        self.use_temporal_enhancement = config.get('use_temporal_enhancement', True)
        
    def decode_i_frames(self, i_frame_latents: torch.Tensor) -> torch.Tensor:
        """
        Decode I-frames from latent representations.
        
        Args:
            i_frame_latents: [B, N, latent_dim] I-frame latents
            
        Returns:
            I-frames: [B, N, C, H, W] decoded I-frames
        """
        B, N, latent_dim = i_frame_latents.shape
        
        # Flatten for batch processing
        latents_flat = i_frame_latents.view(B * N, latent_dim)
        
        # Decode
        i_frames_flat = self.i_frame_decoder(latents_flat)
        
        # Reshape back
        C, H, W = i_frames_flat.shape[1:]
        i_frames = i_frames_flat.view(B, N, C, H, W)
        
        return i_frames
    
    def interpolate_frames(
        self,
        i_frames: torch.Tensor,
        i_frame_indices: List[int],
        total_frames: int,
        semantic_desc: Dict
    ) -> torch.Tensor:
        """
        Generate all frames by interpolating between I-frames.
        
        Args:
            i_frames: [B, N, C, H, W] decoded I-frames
            i_frame_indices: List of I-frame indices
            total_frames: Total number of frames to generate
            semantic_desc: Semantic description with motion vectors
            
        Returns:
            All frames [B, T, C, H, W]
        """
        B, N, C, H, W = i_frames.shape
        device = i_frames.device
        
        # Prepare output tensor
        all_frames = torch.zeros(B, total_frames, C, H, W, device=device)
        
        # Get semantic embedding and motion vectors
        semantic_embedding = semantic_desc['semantic_embedding']  # [B, description_dim]
        motion_vectors = semantic_desc['motion_vectors']  # [B, T, 2]
        
        # Place I-frames
        for i, idx in enumerate(i_frame_indices):
            all_frames[:, idx, :, :, :] = i_frames[:, i, :, :, :]
        
        # Interpolate between I-frames
        for i in range(len(i_frame_indices) - 1):
            start_idx = i_frame_indices[i]
            end_idx = i_frame_indices[i + 1]
            
            ref_frame = i_frames[:, i, :, :, :]  # Reference I-frame
            
            # Generate intermediate frames
            for frame_idx in range(start_idx + 1, end_idx):
                # Get motion vector for this frame
                motion = motion_vectors[:, frame_idx, :]
                
                # Generate frame
                generated = self.video_generator(
                    ref_frame=ref_frame,
                    semantic_desc=semantic_embedding,
                    motion_vector=motion
                )
                
                all_frames[:, frame_idx, :, :, :] = generated
        
        return all_frames
    
    def forward(self, compressed_data: Dict) -> torch.Tensor:
        """
        Main decoding forward pass.
        
        Args:
            compressed_data: Compressed representation from EncodingAgent
            
        Returns:
            Reconstructed video frames [B, T, C, H, W]
        """
        logger.info(f"🎬 Decoding video...")
        
        # Extract data
        i_frame_latents = compressed_data['i_frame_compressed']['latents']
        i_frame_indices = compressed_data['i_frame_indices']
        total_frames = compressed_data['metadata']['num_frames']
        semantic_desc = compressed_data['semantic_description']
        
        # Step 1: Decode I-frames
        logger.info(f"  🎞️  Decoding {i_frame_latents.shape[1]} I-frames...")
        i_frames = self.decode_i_frames(i_frame_latents)
        
        # Step 2: Generate all frames by interpolation
        logger.info(f"  🎨 Generating {total_frames} frames...")
        all_frames = self.interpolate_frames(
            i_frames=i_frames,
            i_frame_indices=i_frame_indices,
            total_frames=total_frames,
            semantic_desc=semantic_desc
        )
        
        # Step 3: Temporal enhancement (optional)
        if self.use_temporal_enhancement:
            logger.info(f"  ✨ Enhancing temporal consistency...")
            all_frames = self.temporal_enhancer(all_frames)
        
        logger.info(f"  ✅ Decoding complete")
        
        return all_frames


def estimate_decoder_tops(decoder: DecodingAgent, input_shape: Tuple) -> float:
    """
    Estimate TOPS (Tera Operations Per Second) for decoder.
    
    Args:
        decoder: DecodingAgent model
        input_shape: Input shape for compressed data
        
    Returns:
        Estimated TOPS per frame
    """
    try:
        from thop import profile
        
        # Create dummy input
        dummy_compressed_data = {
            'i_frame_compressed': {
                'latents': torch.randn(1, 10, 512)  # 10 I-frames
            },
            'i_frame_indices': list(range(0, 300, 30)),
            'metadata': {
                'num_frames': 300
            },
            'semantic_description': {
                'semantic_embedding': torch.randn(1, 256),
                'motion_vectors': torch.randn(1, 300, 2)
            }
        }
        
        # Profile
        macs, params = profile(decoder, inputs=(dummy_compressed_data,))
        
        # Convert MACs to TOPS per frame
        # MACs = Multiply-Accumulate operations (2 ops each)
        # TOPS = (MACs * 2) / 10^12
        num_frames = 300
        tops_total = (macs * 2) / 1e12
        tops_per_frame = tops_total / num_frames
        
        logger.info(f"  📊 Decoder compute analysis:")
        logger.info(f"     Total MACs: {macs / 1e9:.2f} G")
        logger.info(f"     Parameters: {params / 1e6:.2f} M")
        logger.info(f"     TOPS per frame: {tops_per_frame:.4f}")
        logger.info(f"     TOPS for 30 FPS: {tops_per_frame * 30:.2f}")
        
        return tops_per_frame
        
    except ImportError:
        logger.warning("thop not installed - cannot estimate TOPS")
        return 0.0


# ============================================================
# MAIN DECOMPRESSION ENTRY POINT
# ============================================================

def decompress_video_tensor(
    compressed_data: Dict,
    config: Dict,
    device: str = 'cuda'
) -> Dict:
    """
    Decompress video using DecodingAgent.
    
    Args:
        compressed_data: Compressed representation from EncodingAgent
        config: Configuration dict
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        {
            'reconstructed_frames': [B, T, C, H, W] tensor,
            'decode_time_seconds': decoding time,
            'tops_per_frame': estimated TOPS usage
        }
    """
    import time
    
    # Move compressed data to device
    def move_to_device(obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [move_to_device(item, device) if isinstance(item, torch.Tensor) else item for item in obj]
        return obj
    
    compressed_data = move_to_device(compressed_data, device)
    
    # Create decoding agent
    decoder = DecodingAgent(config).to(device)
    decoder.eval()
    
    # Estimate TOPS before decoding
    tops_per_frame = estimate_decoder_tops(decoder, input_shape=None)
    
    # Decode
    start_time = time.time()
    with torch.no_grad():
        reconstructed_frames = decoder(compressed_data)
    decode_time = time.time() - start_time
    
    num_frames = reconstructed_frames.shape[1]
    fps = num_frames / decode_time
    
    logger.info(f"  ⏱️  Decoding performance:")
    logger.info(f"     Time: {decode_time:.2f}s")
    logger.info(f"     FPS: {fps:.1f}")
    logger.info(f"     TOPS/frame: {tops_per_frame:.4f}")
    
    return {
        'reconstructed_frames': reconstructed_frames,
        'decode_time_seconds': decode_time,
        'decode_fps': fps,
        'tops_per_frame': tops_per_frame,
        'tops_at_30fps': tops_per_frame * 30
    }


