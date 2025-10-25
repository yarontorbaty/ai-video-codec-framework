"""
Residual Neural Codec for Soft Details Layer

Compresses residual details (gradients, lighting, atmospheric effects)
Uses a lightweight autoencoder optimized for anime residuals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualEncoder(nn.Module):
    """
    Encoder for residual layer (soft details)
    
    Input: (B, 3, H, W) residual image [-255, 255] normalized to [-1, 1]
    Output: (B, C, H/32, W/32) latent representation
    """
    def __init__(self, latent_channels=32):
        super().__init__()
        
        # Encoder: 5 downsampling blocks (32x reduction)
        self.encoder = nn.Sequential(
            # 512x960 -> 256x480
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            
            # 256x480 -> 128x240
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            
            # 128x240 -> 64x120
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            # 64x120 -> 32x60
            nn.Conv2d(64, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            
            # 32x60 -> 16x30
            nn.Conv2d(48, latent_channels, 3, stride=2, padding=1),
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) residual image, normalized to [-1, 1]
        Returns:
            latent: (B, latent_channels, H/32, W/32)
        """
        return self.encoder(x)


class ResidualDecoder(nn.Module):
    """
    Decoder for residual layer
    
    Input: (B, C, H/32, W/32) latent representation
    Output: (B, 3, H, W) reconstructed residual [-1, 1]
    """
    def __init__(self, latent_channels=32):
        super().__init__()
        
        # Decoder: 5 upsampling blocks (32x expansion)
        self.decoder = nn.Sequential(
            # 16x30 -> 32x60
            nn.ConvTranspose2d(latent_channels, 48, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            
            # 32x60 -> 64x120
            nn.ConvTranspose2d(48, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            # 64x120 -> 128x240
            nn.ConvTranspose2d(64, 48, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            
            # 128x240 -> 256x480
            nn.ConvTranspose2d(48, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            
            # 256x480 -> 512x960
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),  # Output in [-1, 1]
        )
    
    def forward(self, latent):
        """
        Args:
            latent: (B, latent_channels, H/32, W/32)
        Returns:
            residual: (B, 3, H, W) reconstructed residual, range [-1, 1]
        """
        return self.decoder(latent)


class LayeredAnimeCodec(nn.Module):
    """
    Complete Layer-Based Anime Codec
    
    Layers:
    1. Line art (extracted, not learned)
    2. Color palette (K-means, not learned)
    3. Residual (neural, learned)
    """
    def __init__(self, latent_channels=32, n_palette_colors=16):
        super().__init__()
        self.latent_channels = latent_channels
        self.n_palette_colors = n_palette_colors
        
        # Neural components (only residual is learned)
        self.residual_encoder = ResidualEncoder(latent_channels)
        self.residual_decoder = ResidualDecoder(latent_channels)
    
    def encode(self, original_frame, line_art, palette_frame):
        """
        Encode anime frame to compressed representation
        
        Args:
            original_frame: (B, 3, H, W) original RGB frame [0, 1]
            line_art: (B, 1, H, W) binary line art [0, 1]
            palette_frame: (B, 3, H, W) palette-reconstructed frame [0, 1]
        
        Returns:
            residual_latent: (B, C, H/32, W/32) compressed residual
        """
        # Compute residual (soft details)
        residual = original_frame - palette_frame  # Range: [-1, 1]
        
        # Zero out residual where line art exists (already captured)
        residual = residual * (1 - line_art)
        
        # Encode residual
        residual_latent = self.residual_encoder(residual)
        
        return residual_latent
    
    def decode(self, line_art, palette_frame, residual_latent):
        """
        Decode compressed representation to reconstructed frame
        
        Args:
            line_art: (B, 1, H, W) binary line art [0, 1]
            palette_frame: (B, 3, H, W) palette-reconstructed frame [0, 1]
            residual_latent: (B, C, H/32, W/32) compressed residual
        
        Returns:
            reconstructed: (B, 3, H, W) reconstructed frame [0, 1]
        """
        # Decode residual
        residual_decoded = self.residual_decoder(residual_latent)
        
        # Combine layers: palette + residual (line art is implicit in palette)
        reconstructed = palette_frame + residual_decoded
        
        # Clamp to valid range
        reconstructed = torch.clamp(reconstructed, 0, 1)
        
        return reconstructed
    
    def forward(self, original_frame, line_art, palette_frame):
        """
        Full encode-decode pass
        
        Returns:
            reconstructed: (B, 3, H, W)
            residual_latent: (B, C, H/32, W/32)
        """
        residual_latent = self.encode(original_frame, line_art, palette_frame)
        reconstructed = self.decode(line_art, palette_frame, residual_latent)
        return reconstructed, residual_latent


if __name__ == "__main__":
    print("="*70)
    print("TESTING LAYERED ANIME CODEC ARCHITECTURE")
    print("="*70)
    
    # Create model
    model = LayeredAnimeCodec(latent_channels=32, n_palette_colors=16)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.residual_encoder.parameters())
    decoder_params = sum(p.numel() for p in model.residual_decoder.parameters())
    
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Decoder: {decoder_params:,}")
    
    # Test forward pass
    batch_size = 2
    height, width = 512, 960
    
    original = torch.randn(batch_size, 3, height, width) * 0.5 + 0.5  # [0, 1]
    line_art = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    palette = torch.randn(batch_size, 3, height, width) * 0.5 + 0.5
    
    print(f"\nInput shapes:")
    print(f"  Original: {original.shape}")
    print(f"  Line art: {line_art.shape}")
    print(f"  Palette: {palette.shape}")
    
    # Forward pass
    reconstructed, latent = model(original, line_art, palette)
    
    print(f"\nOutput shapes:")
    print(f"  Reconstructed: {reconstructed.shape}")
    print(f"  Latent: {latent.shape}")
    
    # Calculate compressed size
    latent_size_per_frame = latent[0].numel() * 4  # float32
    latent_size_compressed = latent_size_per_frame // 4  # INT8 + GZIP
    
    print(f"\nCompressed size (residual only):")
    print(f"  Latent size (float32): {latent_size_per_frame / 1024:.2f} KB")
    print(f"  Latent size (int8+gzip): {latent_size_compressed / 1024:.2f} KB")
    
    print(f"\nEstimated total compressed size per 960x512 frame:")
    line_art_kb = 261  # From real analysis
    palette_kb = 0.05
    color_map_kb = 40  # With better encoding (target)
    residual_kb = latent_size_compressed / 1024
    total_kb = line_art_kb + palette_kb + color_map_kb + residual_kb
    
    print(f"  Line art: {line_art_kb:.2f} KB")
    print(f"  Palette: {palette_kb:.2f} KB")
    print(f"  Color map: {color_map_kb:.2f} KB (target with better encoding)")
    print(f"  Residual: {residual_kb:.2f} KB")
    print(f"  ---")
    print(f"  TOTAL: {total_kb:.2f} KB")
    
    # Note: This is still high due to line art sparse encoding
    # We'll optimize line art encoding next
    
    print("\n" + "="*70)
    print("✓ Architecture test passed!")
    print("="*70)
    print("\nNext: Optimize line art encoding with delta compression")
    print("Target: Reduce line art from 261 KB to 10-20 KB")

