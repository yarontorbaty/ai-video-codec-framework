#!/usr/bin/env python3
"""
True Autoencoder for Image Compression - COMPACT + BATCHNORM
Key fix: Replace GroupNorm with BatchNorm2d (better gradient flow)
Model size: ~200K params (same as iteration_001, but with BatchNorm)
"""

import torch
import torch.nn as nn


class CompressionAutoencoder(nn.Module):
    """
    Compact autoencoder with BatchNorm (the key fix from iteration_002)
    
    Architecture:
    - Input: RGB image (3, H, W)
    - Encoder: 3 → 48 → 64 → 64 → 48 → 32 (5 downsampling stages)
    - Decoder: 32 → 48 → 64 → 64 → 48 → 3 (5 upsampling stages)
    - Latent: (32, H/32, W/32)
    - Size: ~200K params (fits in 22GB VRAM with batch_size=8)
    """
    
    def __init__(self, latent_channels=32):
        super().__init__()
        self.latent_channels = latent_channels
        
        # ENCODER: Same structure as iteration_001, but with BatchNorm instead of GroupNorm
        # Stage 1: 3 → 48, downsample 2x
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(48),  # ← KEY FIX: BatchNorm instead of GroupNorm
            nn.SiLU(),
        )
        
        # Stage 2: 48 → 64, downsample 2x
        self.enc2 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),  # ← KEY FIX
            nn.SiLU(),
        )
        
        # Stage 3: 64 → 64, downsample 2x
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),  # ← KEY FIX
            nn.SiLU(),
        )
        
        # Stage 4: 64 → 48, downsample 2x
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),  # ← KEY FIX
            nn.SiLU(),
        )
        
        # Stage 5: 48 → 32, downsample 2x
        self.enc5 = nn.Sequential(
            nn.Conv2d(48, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),  # ← KEY FIX
            nn.SiLU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=1),
        )
        
        # DECODER: Same structure, but with BatchNorm and bilinear upsampling
        # Stage 1: 32 → 48, upsample 2x
        self.dec1 = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(latent_channels, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),  # ← KEY FIX
            nn.SiLU(),
        )
        
        # Stage 2: 48 → 64, upsample 2x
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # ← KEY FIX
            nn.SiLU(),
        )
        
        # Stage 3: 64 → 64, upsample 2x
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # ← KEY FIX
            nn.SiLU(),
        )
        
        # Stage 4: 64 → 48, upsample 2x
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),  # ← KEY FIX
            nn.SiLU(),
        )
        
        # Stage 5: 48 → 3, upsample 2x (final output)
        self.dec5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(48, 3, kernel_size=5, padding=2),
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
        x = self.dec1(latent)
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


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features
    Not used by default (--no-perceptual flag), but kept for compatibility
    """
    def __init__(self):
        super().__init__()
        # Dummy implementation - not actually used
        pass
    
    def forward(self, x, y):
        return torch.tensor(0.0, device=x.device)


if __name__ == "__main__":
    # Test the architecture
    model = CompressionAutoencoder(latent_channels=32)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"✓ Target: ~200K params (should fit with batch_size=8)")
    
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
    target = torch.rand(2, 3, 512, 960)
    
    for step in range(100):
        recon, _ = model(target)
        loss = torch.nn.functional.mse_loss(recon, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0 or step == 99:
            print(f"Step {step:3d}: Loss = {loss.item():.6f}")
    
    if loss.item() < 0.01:
        print("\n✅ SUCCESS: Model CAN learn")
    elif loss.item() < 0.05:
        print(f"\n✅ OK: Model learning (loss = {loss.item():.6f})")
    else:
        print(f"\n⚠️ SLOW: Model learning slowly (loss = {loss.item():.6f})")
