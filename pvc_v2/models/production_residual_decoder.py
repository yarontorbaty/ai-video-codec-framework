"""
Production Residual Decoder - Enhanced for 28-30 dB PSNR

Matches ProductionResidualEncoder with skip connections.
Target: ~60M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import AttentionBlock from encoder
try:
    from pvc_v2.models.production_residual_encoder import AttentionBlock
except ImportError:
    # Fallback for when running as part of package
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.production_residual_encoder import AttentionBlock


class UpsampleBlock(nn.Module):
    """Upsampling block with skip connection."""
    
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=False):
        super().__init__()
        
        # Upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Combine with skip
        self.conv_skip = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        self.conv_main = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Refinement
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Optional attention
        if use_attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = None
    
    def forward(self, x, skip):
        # Upsample main path
        x = self.upsample(x)
        x = self.conv_main(x)
        
        # Process skip connection
        skip = self.conv_skip(skip)
        
        # Ensure same spatial size
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # Concatenate and refine
        combined = torch.cat([x, skip], dim=1)
        out = self.refine(combined)
        
        # Apply attention if present
        if self.attention is not None:
            out = self.attention(out)
        
        return out


class ProductionResidualDecoder(nn.Module):
    """
    Production-quality residual decoder.
    
    Architecture:
    - Input: Compressed latent (4x4x128)
    - 6 decoding blocks with decreasing channels: 1024, 1024, 768, 512, 256, 128
    - Skip connections from encoder
    - Attention at blocks 1, 2, 3
    - Output: 256x256x3 (reconstructed residual)
    
    Parameters: ~60M
    """
    
    def __init__(self, latent_dim=128, out_channels=3):
        super().__init__()
        
        # Dequantization
        self.dequant_conv = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)
        
        # From latent to feature space
        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_dim, 640, kernel_size=3, padding=1),
            nn.GroupNorm(8, 640),
            nn.ReLU(inplace=True)
        )
        
        # Decoding blocks (5 blocks, matching encoder)
        self.up5 = UpsampleBlock(640, 640, 512, use_attention=True)     # 4x4 -> 8x8
        self.up4 = UpsampleBlock(512, 640, 384, use_attention=True)     # 8x8 -> 16x16
        self.up3 = UpsampleBlock(384, 512, 256, use_attention=False)    # 16x16 -> 32x32
        self.up2 = UpsampleBlock(256, 256, 128, use_attention=False)    # 32x32 -> 64x64
        self.up1 = UpsampleBlock(128, 128, 64, use_attention=False)     # 64x64 -> 128x128
        
        # Final upsampling to original resolution
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Final conv to output
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=7, padding=3),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, latent, skips):
        """
        Args:
            latent: Compressed latent (B, latent_dim, 4, 4)
            skips: Tuple of skip connections from encoder
                   (feat1, feat2, feat3, feat4, feat5)
        """
        feat1, feat2, feat3, feat4, feat5 = skips
        
        # Dequantize
        x = self.dequant_conv(latent)
        x = self.from_latent(x)  # 4x4, 640 channels
        
        # Decode with skip connections
        x = self.up5(x, feat5)  # 8x8, 512 channels
        x = self.up4(x, feat4)  # 16x16, 384 channels
        x = self.up3(x, feat3)  # 32x32, 256 channels
        x = self.up2(x, feat2)  # 64x64, 128 channels
        x = self.up1(x, feat1)  # 128x128, 64 channels
        
        # Final upsampling and output
        x = self.final_up(x)  # 256x256, 64 channels
        out = self.conv_out(x)  # 256x256, 3 channels
        
        return out
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    from production_residual_encoder import ProductionResidualEncoder
    
    # Test encoder-decoder pair
    encoder = ProductionResidualEncoder()
    decoder = ProductionResidualDecoder()
    
    # Count parameters
    enc_params = encoder.get_num_parameters()
    dec_params = decoder.get_num_parameters()
    total_params = enc_params + dec_params
    
    print(f"Encoder Parameters: {enc_params / 1e6:.2f}M")
    print(f"Decoder Parameters: {dec_params / 1e6:.2f}M")
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
    print(f"vs Baseline: 32.4M → {total_params / 1e6:.2f}M ({total_params / 32.4e6:.1f}x)")
    
    # Test forward pass
    x = torch.randn(1, 3, 256, 256)
    
    # Encode
    latent, skips = encoder(x)
    print(f"\nLatent shape: {latent.shape}")
    
    # Decode
    recon = decoder(latent, skips)
    print(f"Reconstruction shape: {recon.shape}")
    
    # Check output range
    print(f"Output range: [{recon.min():.2f}, {recon.max():.2f}]")

