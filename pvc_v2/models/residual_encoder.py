#!/usr/bin/env python3
"""
Residual Encoder for Hybrid PVC v2.0

Compresses residuals (original - coarse reconstruction) using:
- Lightweight CNN for feature extraction
- DCT transformation for frequency domain compression
- Quantization for size reduction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualEncoder(nn.Module):
    """
    Encode residuals into compressed representation.
    
    Architecture:
    - 3 conv layers (3→32→16→8 channels)
    - DCT transformation (8x8 blocks)
    - Quantization (adjustable Q parameter)
    
    Input: Residual image (B, 3, H, W) in [-1, 1] range
    Output: Compressed coefficients
    """
    
    def __init__(self, quality_factor=20):
        """
        Initialize residual encoder.
        
        Args:
            quality_factor: Quantization parameter (10-50)
                           Lower = more compression, less quality
                           Higher = less compression, more quality
        """
        super(ResidualEncoder, self).__init__()
        
        self.quality_factor = quality_factor
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(8)
        
        # DCT quantization matrix (similar to JPEG)
        self.register_buffer('dct_quant_matrix', self._init_quant_matrix())
    
    def _init_quant_matrix(self):
        """Initialize DCT quantization matrix (8x8)."""
        # Standard JPEG luminance quantization matrix
        base_matrix = torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float32)
        
        return base_matrix
    
    def dct_2d(self, x):
        """
        Apply 2D DCT to 8x8 blocks using manual implementation.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            DCT coefficients (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Ensure dimensions are divisible by 8
        assert H % 8 == 0 and W % 8 == 0, "Image dimensions must be divisible by 8"
        
        # For simplicity, use FFT-based approximation
        # In production, use scipy.fftpack.dct or opencv
        
        # Reshape to 8x8 blocks
        x = x.view(B, C, H // 8, 8, W // 8, 8)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()  # (B, C, H//8, W//8, 8, 8)
        
        # Use 2D FFT as approximation (real part approximates DCT)
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        dct_blocks = x_fft.real
        
        # Reshape back
        dct_blocks = dct_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        dct_blocks = dct_blocks.view(B, C, H, W)
        
        return dct_blocks
    
    def quantize(self, dct_coeffs):
        """
        Quantize DCT coefficients.
        
        Args:
            dct_coeffs: DCT coefficients (B, C, H, W)
        
        Returns:
            Quantized coefficients
        """
        B, C, H, W = dct_coeffs.shape
        
        # Scale quantization matrix by quality factor
        quant_matrix = self.dct_quant_matrix * (self.quality_factor / 10.0)
        
        # Expand to match input dimensions
        quant_matrix = quant_matrix.unsqueeze(0).unsqueeze(0)  # (1, 1, 8, 8)
        quant_matrix = quant_matrix.repeat(1, 1, H // 8, W // 8)  # (1, 1, H, W)
        
        # Quantize
        quantized = torch.round(dct_coeffs / quant_matrix)
        
        return quantized, quant_matrix
    
    def forward(self, residual):
        """
        Encode residual image.
        
        Args:
            residual: Residual image (B, 3, H, W) in [-1, 1] range
        
        Returns:
            Compressed representation dict with:
            - features: CNN features (B, 8, H, W)
            - quantized_dct: Quantized DCT coefficients
            - quant_matrix: Quantization matrix used
        """
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(residual)))
        x = F.relu(self.bn2(self.conv2(x)))
        features = F.relu(self.bn3(self.conv3(x)))  # (B, 8, H, W)
        
        # Apply DCT to features
        dct_coeffs = self.dct_2d(features)
        
        # Quantize DCT coefficients
        quantized_dct, quant_matrix = self.quantize(dct_coeffs)
        
        return {
            'features': features,  # For skip connections if needed
            'quantized_dct': quantized_dct,
            'quant_matrix': quant_matrix
        }
    
    def get_compressed_size(self, compressed_data):
        """
        Estimate compressed size in bytes.
        
        Args:
            compressed_data: Output from forward()
        
        Returns:
            Estimated size in bytes
        """
        quantized = compressed_data['quantized_dct']
        
        # Count non-zero coefficients (these would be entropy coded)
        non_zero = torch.count_nonzero(quantized)
        
        # Estimate: 2 bytes per non-zero coefficient (position + value)
        estimated_size = non_zero.item() * 2
        
        return estimated_size


if __name__ == "__main__":
    # Test the encoder
    print("Testing Residual Encoder...")
    
    encoder = ResidualEncoder(quality_factor=20)
    
    # Create dummy residual (256x256x3)
    residual = torch.randn(1, 3, 256, 256) * 0.5  # [-0.5, 0.5] range
    
    # Encode
    compressed = encoder(residual)
    
    print(f"Input shape: {residual.shape}")
    print(f"Features shape: {compressed['features'].shape}")
    print(f"Quantized DCT shape: {compressed['quantized_dct'].shape}")
    print(f"Estimated compressed size: {encoder.get_compressed_size(compressed)} bytes")
    
    # Calculate compression ratio
    original_size = residual.numel() * 4  # 4 bytes per float32
    compressed_size = encoder.get_compressed_size(compressed)
    compression_ratio = (1 - compressed_size / original_size) * 100
    
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {compression_ratio:.1f}%")
    print("\n✅ Residual Encoder working!")

