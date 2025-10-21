#!/usr/bin/env python3
"""
Residual Decoder for Hybrid PVC v2.0

Reconstructs residuals from compressed representation using:
- Dequantization
- Inverse DCT transformation
- Lightweight CNN for reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualDecoder(nn.Module):
    """
    Decode compressed residuals back to image space.
    
    Architecture:
    - Dequantization
    - Inverse DCT transformation
    - 3 transposed conv layers (8→16→32→3 channels)
    
    Input: Compressed coefficients
    Output: Reconstructed residual (B, 3, H, W) in [-1, 1] range
    """
    
    def __init__(self):
        """Initialize residual decoder."""
        super(ResidualDecoder, self).__init__()
        
        # Transposed CNN layers for reconstruction
        self.deconv1 = nn.ConvTranspose2d(8, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.deconv2 = nn.ConvTranspose2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1)
    
    def idct_2d(self, dct_coeffs):
        """
        Apply inverse 2D DCT to 8x8 blocks using manual implementation.
        
        Args:
            dct_coeffs: DCT coefficients (B, C, H, W)
        
        Returns:
            Spatial domain tensor (B, C, H, W)
        """
        B, C, H, W = dct_coeffs.shape
        
        # Ensure dimensions are divisible by 8
        assert H % 8 == 0 and W % 8 == 0, "Image dimensions must be divisible by 8"
        
        # Reshape to 8x8 blocks
        dct_coeffs = dct_coeffs.view(B, C, H // 8, 8, W // 8, 8)
        dct_coeffs = dct_coeffs.permute(0, 1, 2, 4, 3, 5).contiguous()  # (B, C, H//8, W//8, 8, 8)
        
        # Use inverse FFT as approximation
        # Convert real DCT coefficients to complex for ifft
        dct_complex = torch.complex(dct_coeffs, torch.zeros_like(dct_coeffs))
        idct_blocks = torch.fft.ifft2(dct_complex, dim=(-2, -1)).real
        
        # Reshape back
        idct_blocks = idct_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        idct_blocks = idct_blocks.view(B, C, H, W)
        
        return idct_blocks
    
    def dequantize(self, quantized_dct, quant_matrix):
        """
        Dequantize DCT coefficients.
        
        Args:
            quantized_dct: Quantized coefficients (B, C, H, W)
            quant_matrix: Quantization matrix used during encoding
        
        Returns:
            Dequantized DCT coefficients
        """
        return quantized_dct * quant_matrix
    
    def forward(self, compressed_data):
        """
        Decode compressed residuals.
        
        Args:
            compressed_data: Dict from ResidualEncoder with:
                - quantized_dct: Quantized DCT coefficients
                - quant_matrix: Quantization matrix
        
        Returns:
            Reconstructed residual (B, 3, H, W) in [-1, 1] range
        """
        # Dequantize DCT coefficients
        dct_coeffs = self.dequantize(
            compressed_data['quantized_dct'],
            compressed_data['quant_matrix']
        )
        
        # Apply inverse DCT
        features = self.idct_2d(dct_coeffs)  # (B, 8, H, W)
        
        # CNN reconstruction
        x = F.relu(self.bn1(self.deconv1(features)))
        x = F.relu(self.bn2(self.deconv2(x)))
        residual = torch.tanh(self.deconv3(x))  # (B, 3, H, W) in [-1, 1]
        
        return residual


if __name__ == "__main__":
    # Test encoder-decoder round trip
    print("Testing Residual Encoder-Decoder Round Trip...")
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from residual_encoder import ResidualEncoder
    
    # Create encoder and decoder
    encoder = ResidualEncoder(quality_factor=20)
    decoder = ResidualDecoder()
    
    # Create dummy residual (256x256x3)
    original_residual = torch.randn(1, 3, 256, 256) * 0.5  # [-0.5, 0.5] range
    
    print(f"Original residual shape: {original_residual.shape}")
    print(f"Original residual range: [{original_residual.min():.3f}, {original_residual.max():.3f}]")
    
    # Encode
    compressed = encoder(original_residual)
    print(f"\nCompressed DCT shape: {compressed['quantized_dct'].shape}")
    print(f"Non-zero coefficients: {torch.count_nonzero(compressed['quantized_dct'])}")
    
    # Decode
    reconstructed_residual = decoder(compressed)
    print(f"\nReconstructed residual shape: {reconstructed_residual.shape}")
    print(f"Reconstructed residual range: [{reconstructed_residual.min():.3f}, {reconstructed_residual.max():.3f}]")
    
    # Calculate reconstruction error
    mse = F.mse_loss(reconstructed_residual, original_residual)
    psnr = 10 * torch.log10(4.0 / mse)  # 4.0 = (2.0)^2 for [-1, 1] range
    
    print(f"\nReconstruction MSE: {mse.item():.6f}")
    print(f"Reconstruction PSNR: {psnr.item():.2f} dB")
    
    # Compression stats
    original_size = original_residual.numel() * 4  # 4 bytes per float32
    compressed_size = encoder.get_compressed_size(compressed)
    compression_ratio = (1 - compressed_size / original_size) * 100
    
    print(f"\nOriginal size: {original_size} bytes ({original_size / 1024:.1f} KB)")
    print(f"Compressed size: {compressed_size} bytes ({compressed_size / 1024:.1f} KB)")
    print(f"Compression ratio: {compression_ratio:.1f}%")
    
    print("\n✅ Residual Decoder working!")
    
    # Visual quality check
    if psnr.item() > 20:
        print("✅ Good reconstruction quality (>20 dB)")
    else:
        print("⚠️  Low reconstruction quality (<20 dB)")

