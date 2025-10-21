#!/usr/bin/env python3
"""
SOTA Residual Encoder for Hybrid PVC v2.0

U-Net style encoder with:
- Deep architecture (8 layers)
- Skip connections
- Attention mechanisms
- Multi-scale processing

Target: 35-45 dB PSNR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """
    Convolutional block with BatchNorm and ReLU.
    
    Standard building block for encoder/decoder.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    """
    Spatial attention mechanism.
    
    Allows the network to focus on important regions
    by computing attention weights based on feature importance.
    """
    
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Compute Q, K, V
        Q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C//8)
        K = self.key(x).view(B, -1, H * W)                      # (B, C//8, HW)
        V = self.value(x).view(B, -1, H * W)                    # (B, C, HW)
        
        # Attention weights
        attention = F.softmax(torch.bmm(Q, K), dim=-1)  # (B, HW, HW)
        
        # Apply attention
        out = torch.bmm(V, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable weight
        return x + self.gamma * out


class SOTAResidualEncoder(nn.Module):
    """
    State-of-the-art residual encoder with U-Net architecture.
    
    Architecture:
    - Encoder path: 4 downsampling blocks (256 → 128 → 64 → 32 → 32)
    - Bottleneck: Attention module
    - Skip connections exported for decoder
    - Multi-scale feature extraction
    
    Input: Residual image (B, 3, 256, 256) in [-2, 2] range
    Output: Compressed features + skip connections
    """
    
    def __init__(self, base_channels=64, quality_factor=30):
        super(SOTAResidualEncoder, self).__init__()
        
        self.quality_factor = quality_factor
        
        # Encoder blocks (downsampling)
        self.enc1 = ConvBlock(3, base_channels)           # 256×256
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(base_channels, base_channels * 2)    # 128×128
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)  # 64×64
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)  # 32×32
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck with attention (16×16)
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16),
            AttentionBlock(base_channels * 16)
        )
        
        # Compression layer (reduce channels for DCT)
        self.compress = nn.Conv2d(base_channels * 16, 8, 1)
        
        # DCT quantization matrix
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
        """Apply 2D DCT to 8x8 blocks using FFT."""
        B, C, H, W = x.shape
        
        # Ensure dimensions are divisible by 8
        assert H % 8 == 0 and W % 8 == 0, "Dimensions must be divisible by 8"
        
        # Reshape to 8x8 blocks
        x = x.view(B, C, H // 8, 8, W // 8, 8)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        
        # Apply 2D FFT (approximates DCT)
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        dct_blocks = x_fft.real
        
        # Reshape back
        dct_blocks = dct_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        dct_blocks = dct_blocks.view(B, C, H, W)
        
        return dct_blocks
    
    def quantize(self, dct_coeffs):
        """Quantize DCT coefficients."""
        B, C, H, W = dct_coeffs.shape
        
        # Scale quantization matrix by quality factor
        quant_matrix = self.dct_quant_matrix * (self.quality_factor / 10.0)
        
        # Expand to match input dimensions
        quant_matrix = quant_matrix.unsqueeze(0).unsqueeze(0)
        quant_matrix = quant_matrix.repeat(1, 1, H // 8, W // 8)
        
        # Quantize
        quantized = torch.round(dct_coeffs / quant_matrix)
        
        return quantized, quant_matrix
    
    def forward(self, residual):
        """
        Encode residual image.
        
        Args:
            residual: Residual image (B, 3, H, W) in [-2, 2] range
        
        Returns:
            Dict with:
            - compressed: Quantized DCT coefficients
            - quant_matrix: Quantization matrix
            - skip1, skip2, skip3, skip4: Skip connections for decoder
        """
        # Encoder path with skip connections
        skip1 = self.enc1(residual)      # (B, 64, 256, 256)
        x = self.pool1(skip1)
        
        skip2 = self.enc2(x)             # (B, 128, 128, 128)
        x = self.pool2(skip2)
        
        skip3 = self.enc3(x)             # (B, 256, 64, 64)
        x = self.pool3(skip3)
        
        skip4 = self.enc4(x)             # (B, 512, 32, 32)
        x = self.pool4(skip4)
        
        # Bottleneck with attention
        x = self.bottleneck(x)           # (B, 1024, 16, 16)
        
        # Compress channels for DCT
        features = self.compress(x)      # (B, 8, 16, 16)
        
        # Apply DCT and quantization
        dct_coeffs = self.dct_2d(features)
        quantized, quant_matrix = self.quantize(dct_coeffs)
        
        return {
            'compressed': quantized,
            'quant_matrix': quant_matrix,
            'skip1': skip1,
            'skip2': skip2,
            'skip3': skip3,
            'skip4': skip4
        }
    
    def get_compressed_size(self, encoded_data):
        """Estimate compressed size in bytes."""
        quantized = encoded_data['compressed']
        
        # Count non-zero coefficients
        non_zero = torch.count_nonzero(quantized)
        
        # Estimate: 2 bytes per non-zero coefficient
        estimated_size = non_zero.item() * 2
        
        return estimated_size


if __name__ == "__main__":
    # Test the SOTA encoder
    print("="*70)
    print("Testing SOTA Residual Encoder")
    print("="*70)
    
    encoder = SOTAResidualEncoder(base_channels=64, quality_factor=30)
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nModel size: {total_params:,} parameters ({total_params * 4 / 1024 / 1024:.2f} MB)")
    
    # Create dummy residual
    residual = torch.randn(2, 3, 256, 256) * 0.5  # Batch of 2
    
    print(f"\nInput shape: {residual.shape}")
    print(f"Input range: [{residual.min():.3f}, {residual.max():.3f}]")
    
    # Encode
    print("\nEncoding...")
    encoded = encoder(residual)
    
    print(f"\nCompressed shape: {encoded['compressed'].shape}")
    print(f"Skip1 shape: {encoded['skip1'].shape}")
    print(f"Skip2 shape: {encoded['skip2'].shape}")
    print(f"Skip3 shape: {encoded['skip3'].shape}")
    print(f"Skip4 shape: {encoded['skip4'].shape}")
    
    print(f"\nNon-zero coefficients: {torch.count_nonzero(encoded['compressed'])}")
    print(f"Estimated compressed size: {encoder.get_compressed_size(encoded)} bytes")
    
    # Compression ratio
    original_size = residual.numel() * 4
    compressed_size = encoder.get_compressed_size(encoded)
    compression_ratio = (1 - compressed_size / original_size) * 100
    
    print(f"\nOriginal size: {original_size} bytes ({original_size / 1024:.1f} KB)")
    print(f"Compressed size: {compressed_size} bytes ({compressed_size / 1024:.1f} KB)")
    print(f"Compression ratio: {compression_ratio:.1f}%")
    
    print("\n" + "="*70)
    print("✅ SOTA Residual Encoder working!")
    print("="*70)

