#!/usr/bin/env python3
"""
True Autoencoder for Image Compression - OPTIMIZED ARCHITECTURE
- Replaced GroupNorm with BatchNorm2d (better gradient flow)
- Larger decoder capacity (can reconstruct from latent alone)
- NO skip connections (pure encode → latent → decode)
- Optimized for high-resolution images (512x960)
"""

import torch
import torch.nn as nn


class CompressionAutoencoder(nn.Module):
    """
    Optimized autoencoder with:
    1. BatchNorm instead of GroupNorm
    2. Larger decoder with more capacity
    3. NO skip connections (pure compression codec)
    4. Better weight initialization
    
    Architecture:
    - Input: RGB image (3, H, W)
    - Encoder: 3 → 48 → 64 → 96 → 64 → 32 (5 downsampling stages = 32x compression)
    - Decoder: 32 → 64 → 96 → 64 → 48 → 3 (5 upsampling stages)
    - Latent: (32, H/32, W/32)
    """
    
    def __init__(self, latent_channels=32):
        super().__init__()
        self.latent_channels = latent_channels
        
        # ENCODER: Progressive downsampling with increasing capacity
        # Stage 1: 3 → 48, downsample 2x (1/2 resolution)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
        )
        
        # Stage 2: 48 → 64, downsample 2x (1/4 resolution)
        self.enc2 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        
        # Stage 3: 64 → 96, downsample 2x (1/8 resolution)
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),
        )
        
        # Stage 4: 96 → 64, downsample 2x (1/16 resolution)
        self.enc4 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        
        # Stage 5: 64 → 32, downsample 2x (1/32 resolution - latent)
        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Conv2d(48, latent_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.SiLU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=1),  # Final compression
        )
        
        # DECODER: Progressive upsampling with larger capacity
        # Initial processing of latent
        self.dec_init = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.SiLU(),
        )
        
        # Stage 1: 32 → 64, upsample 2x (1/16 resolution)
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(latent_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        
        # Stage 2: 64 → 96, upsample 2x (1/8 resolution)
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),
        )
        
        # Stage 3: 96 → 64, upsample 2x (1/4 resolution)
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        
        # Stage 4: 64 → 48, upsample 2x (1/2 resolution)
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
        )
        
        # Stage 5: 48 → 3, upsample 2x (full resolution - output)
        self.dec5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.Conv2d(24, 3, kernel_size=5, padding=2),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        """Encode image to latent representation"""
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        latent = self.enc5(x)
        return latent
    
    def decode(self, latent):
        """Decode latent to image - NO input needed!"""
        x = self.dec_init(latent)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        output = self.dec5(x)
        return output
    
    def forward(self, x):
        """Full forward pass: encode → decode"""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


if __name__ == "__main__":
    # Test the architecture
    model = CompressionAutoencoder(latent_channels=32)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 960)
    print(f"\nInput shape: {x.shape}")
    
    recon, latent = model(x)
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {recon.shape}")
    print(f"Output range: [{recon.min():.4f}, {recon.max():.4f}]")
    
    # Test if model can overfit 1 batch (sanity check)
    print("\n" + "="*60)
    print("SANITY CHECK: Can model overfit 1 batch?")
    print("="*60)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    target = torch.rand(2, 3, 512, 960)  # Random target
    
    for step in range(100):
        recon, _ = model(target)
        loss = torch.nn.functional.mse_loss(recon, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0 or step == 99:
            print(f"Step {step:3d}: Loss = {loss.item():.6f}")
    
    if loss.item() < 0.005:
        print("\n✅ SUCCESS: Model CAN learn (excellent)")
    elif loss.item() < 0.02:
        print(f"\n✅ OK: Model can learn (loss = {loss.item():.6f})")
    else:
        print(f"\n⚠️ SLOW: Model learning slowly (loss = {loss.item():.6f})")
