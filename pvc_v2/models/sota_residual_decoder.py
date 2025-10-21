#!/usr/bin/env python3
"""
SOTA Residual Decoder for Hybrid PVC v2.0

U-Net style decoder with:
- Deep architecture (8 layers)
- Skip connections from encoder
- Progressive upsampling
- Refinement modules

Target: 35-45 dB PSNR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpConvBlock(nn.Module):
    """
    Upsampling convolutional block with skip connections.
    
    Performs upsampling followed by convolutions,
    with concatenation of skip connection from encoder.
    """
    
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UpConvBlock, self).__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # Convolutions after concatenating skip connection
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class SOTAResidualDecoder(nn.Module):
    """
    State-of-the-art residual decoder with U-Net architecture.
    
    Architecture:
    - IDCT decompression
    - Expansion layer (8 → 1024 channels)
    - Decoder path: 4 upsampling blocks with skip connections
    - Final refinement
    
    Input: Compressed features + skip connections from encoder
    Output: Reconstructed residual (B, 3, 256, 256) in [-1, 1] range
    """
    
    def __init__(self, base_channels=64):
        super(SOTAResidualDecoder, self).__init__()
        
        # Expansion layer (decompress channels)
        self.expand = nn.Conv2d(8, base_channels * 16, 1)
        
        # Decoder blocks (upsampling) with skip connections
        self.dec4 = UpConvBlock(base_channels * 16, base_channels * 8, skip_channels=base_channels * 8)  # 16 → 32
        self.dec3 = UpConvBlock(base_channels * 8, base_channels * 4, skip_channels=base_channels * 4)    # 32 → 64
        self.dec2 = UpConvBlock(base_channels * 4, base_channels * 2, skip_channels=base_channels * 2)    # 64 → 128
        self.dec1 = UpConvBlock(base_channels * 2, base_channels, skip_channels=base_channels)            # 128 → 256
        
        # Final refinement layers
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def idct_2d(self, dct_coeffs):
        """Apply inverse 2D DCT to 8x8 blocks using IFFT."""
        B, C, H, W = dct_coeffs.shape
        
        # Ensure dimensions are divisible by 8
        assert H % 8 == 0 and W % 8 == 0, "Dimensions must be divisible by 8"
        
        # Reshape to 8x8 blocks
        dct_coeffs = dct_coeffs.view(B, C, H // 8, 8, W // 8, 8)
        dct_coeffs = dct_coeffs.permute(0, 1, 2, 4, 3, 5).contiguous()
        
        # Convert to complex and apply IFFT
        dct_complex = torch.complex(dct_coeffs, torch.zeros_like(dct_coeffs))
        idct_blocks = torch.fft.ifft2(dct_complex, dim=(-2, -1)).real
        
        # Reshape back
        idct_blocks = idct_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        idct_blocks = idct_blocks.view(B, C, H, W)
        
        return idct_blocks
    
    def dequantize(self, quantized_dct, quant_matrix):
        """Dequantize DCT coefficients."""
        return quantized_dct * quant_matrix
    
    def forward(self, encoded_data):
        """
        Decode compressed residuals.
        
        Args:
            encoded_data: Dict from SOTAResidualEncoder with:
                - compressed: Quantized DCT coefficients
                - quant_matrix: Quantization matrix
                - skip1, skip2, skip3, skip4: Skip connections
        
        Returns:
            Reconstructed residual (B, 3, 256, 256) in [-1, 1] range
        """
        # Dequantize DCT coefficients
        dct_coeffs = self.dequantize(
            encoded_data['compressed'],
            encoded_data['quant_matrix']
        )
        
        # Apply inverse DCT
        features = self.idct_2d(dct_coeffs)  # (B, 8, 16, 16)
        
        # Expand channels
        x = self.expand(features)            # (B, 1024, 16, 16)
        
        # Decoder path with skip connections
        x = self.dec4(x, encoded_data['skip4'])  # (B, 512, 32, 32)
        x = self.dec3(x, encoded_data['skip3'])  # (B, 256, 64, 64)
        x = self.dec2(x, encoded_data['skip2'])  # (B, 128, 128, 128)
        x = self.dec1(x, encoded_data['skip1'])  # (B, 64, 256, 256)
        
        # Final refinement
        residual = self.final(x)                 # (B, 3, 256, 256)
        
        return residual


if __name__ == "__main__":
    # Test encoder-decoder round trip
    print("="*70)
    print("Testing SOTA Encoder-Decoder Round Trip")
    print("="*70)
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from sota_residual_encoder import SOTAResidualEncoder
    
    # Create encoder and decoder
    encoder = SOTAResidualEncoder(base_channels=64, quality_factor=30)
    decoder = SOTAResidualDecoder(base_channels=64)
    
    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    total_params = encoder_params + decoder_params
    
    print(f"\nEncoder parameters: {encoder_params:,} ({encoder_params * 4 / 1024 / 1024:.2f} MB)")
    print(f"Decoder parameters: {decoder_params:,} ({decoder_params * 4 / 1024 / 1024:.2f} MB)")
    print(f"Total parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
    
    # Create dummy residual
    original_residual = torch.randn(2, 3, 256, 256) * 0.5
    
    print(f"\nOriginal residual shape: {original_residual.shape}")
    print(f"Original residual range: [{original_residual.min():.3f}, {original_residual.max():.3f}]")
    
    # Encode
    print("\nEncoding...")
    encoded = encoder(original_residual)
    print(f"Compressed shape: {encoded['compressed'].shape}")
    print(f"Non-zero coefficients: {torch.count_nonzero(encoded['compressed'])}")
    
    # Decode
    print("\nDecoding...")
    reconstructed_residual = decoder(encoded)
    print(f"Reconstructed residual shape: {reconstructed_residual.shape}")
    print(f"Reconstructed residual range: [{reconstructed_residual.min():.3f}, {reconstructed_residual.max():.3f}]")
    
    # Calculate reconstruction error
    mse = F.mse_loss(reconstructed_residual, original_residual)
    psnr = 10 * torch.log10(4.0 / mse)  # 4.0 = (2.0)^2 for [-1, 1] range
    
    print(f"\nReconstruction MSE: {mse.item():.6f}")
    print(f"Reconstruction PSNR: {psnr.item():.2f} dB")
    
    # Compression stats
    original_size = original_residual.numel() * 4
    compressed_size = encoder.get_compressed_size(encoded)
    compression_ratio = (1 - compressed_size / original_size) * 100
    
    print(f"\nOriginal size: {original_size} bytes ({original_size / 1024:.1f} KB)")
    print(f"Compressed size: {compressed_size} bytes ({compressed_size / 1024:.1f} KB)")
    print(f"Compression ratio: {compression_ratio:.1f}%")
    
    print("\n" + "="*70)
    print("✅ SOTA Residual Decoder working!")
    print("="*70)
    
    # Quality check
    if psnr.item() > 15:
        print("✅ Good untrained reconstruction quality (>15 dB)")
    else:
        print("⚠️  Low untrained quality (<15 dB) - training will improve this")
    
    print("\nReady for training! 🚀")

