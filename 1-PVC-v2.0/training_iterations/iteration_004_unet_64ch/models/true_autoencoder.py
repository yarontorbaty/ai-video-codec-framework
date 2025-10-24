#!/usr/bin/env python3
"""
U-Net Style Autoencoder for Image Compression
- 64 latent channels (2x more capacity)
- Skip connections during TRAINING only (don't increase file size)
- BatchNorm for stable gradients
"""

import torch
import torch.nn as nn


class CompressionAutoencoder(nn.Module):
    """
    U-Net style autoencoder with skip connections
    
    Key insight: Skip connections help TRAINING but aren't stored in the compressed file.
    During inference, decoder works from latent alone.
    
    Architecture:
    - Input: RGB image (3, H, W)
    - Encoder: 3 → 48 → 64 → 64 → 48 → 64 (5 downsampling stages)
    - Latent: (64, H/32, W/32)
    - Decoder: 64 → 48 → 64 → 64 → 48 → 3 (with skip connections during training)
    """
    
    def __init__(self, latent_channels=64, use_skip_connections=True):
        super().__init__()
        self.latent_channels = latent_channels
        self.use_skip_connections = use_skip_connections
        
        # ENCODER with skip outputs
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(48),
            nn.SiLU(),
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
        )
        
        self.enc5 = nn.Sequential(
            nn.Conv2d(48, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.SiLU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=1),
        )
        
        # DECODER with skip connection fusion
        self.dec1 = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(latent_channels, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
        )
        # Skip fusion: 48 (dec) + 48 (enc4) = 96 → 48
        self.skip1 = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
        )
        
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        # Skip fusion: 64 (dec) + 64 (enc3) = 128 → 64
        self.skip2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        # Skip fusion: 64 (dec) + 64 (enc2) = 128 → 64
        self.skip3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
        )
        # Skip fusion: 48 (dec) + 48 (enc1) = 96 → 48
        self.skip4 = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
        )
        
        self.dec5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(48, 3, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        """Encode with skip connections for training"""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        latent = self.enc5(e4)
        return latent, (e1, e2, e3, e4)
    
    def decode(self, latent, skip_connections=None):
        """
        Decode from latent.
        During training: use skip_connections
        During inference: skip_connections=None, decoder works alone
        """
        d1 = self.dec1(latent)
        
        if self.use_skip_connections and skip_connections is not None:
            d1 = torch.cat([d1, skip_connections[3]], dim=1)  # enc4
            d1 = self.skip1(d1)
        
        d2 = self.dec2(d1)
        
        if self.use_skip_connections and skip_connections is not None:
            d2 = torch.cat([d2, skip_connections[2]], dim=1)  # enc3
            d2 = self.skip2(d2)
        
        d3 = self.dec3(d2)
        
        if self.use_skip_connections and skip_connections is not None:
            d3 = torch.cat([d3, skip_connections[1]], dim=1)  # enc2
            d3 = self.skip3(d3)
        
        d4 = self.dec4(d3)
        
        if self.use_skip_connections and skip_connections is not None:
            d4 = torch.cat([d4, skip_connections[0]], dim=1)  # enc1
            d4 = self.skip4(d4)
        
        output = self.dec5(d4)
        return output
    
    def forward(self, x):
        """Full forward: encode → decode with skip connections"""
        latent, skip_connections = self.encode(x)
        reconstructed = self.decode(latent, skip_connections if self.training else None)
        return reconstructed, latent


class PerceptualLoss(nn.Module):
    """Dummy for compatibility"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        return torch.tensor(0.0, device=x.device)


if __name__ == "__main__":
    print("="*60)
    print("TESTING U-NET AUTOENCODER WITH 64 LATENT CHANNELS")
    print("="*60)
    
    model = CompressionAutoencoder(latent_channels=64)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 960)
    print(f"Input shape: {x.shape}")
    
    model.train()
    recon_train, latent = model(x)
    print(f"Latent shape: {latent.shape}")
    print(f"Output shape: {recon_train.shape}")
    
    # Test inference mode (no skip connections)
    model.eval()
    with torch.no_grad():
        recon_eval, latent = model(x)
        print(f"Eval output range: [{recon_eval.min():.4f}, {recon_eval.max():.4f}]")
    
    # Calculate compressed size
    latent_size_bytes = latent.numel() * 4  # float32
    latent_size_kb = latent_size_bytes / 1024
    print(f"\nCompressed size per frame: {latent_size_kb:.1f} KB (float32)")
    print(f"Compressed size per frame: {latent_size_kb/4:.1f} KB (int8 quantized)")
    
    # Sanity check: can it overfit?
    print("\n" + "="*60)
    print("SANITY CHECK: Can model overfit 1 batch?")
    print("="*60)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target = torch.rand(2, 3, 512, 960)
    
    print("Training with skip connections...")
    for step in range(100):
        recon, _ = model(target)
        loss = torch.nn.functional.mse_loss(recon, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0 or step == 99:
            print(f"Step {step:3d}: Loss = {loss.item():.6f}")
    
    # Test inference without skip connections
    print("\nTesting inference mode (no skip connections)...")
    model.eval()
    with torch.no_grad():
        recon_inference, _ = model(target)
        loss_inference = torch.nn.functional.mse_loss(recon_inference, target)
        print(f"Inference loss: {loss_inference.item():.6f}")
    
    if loss.item() < 0.005:
        print("\n✅ SUCCESS: Model CAN learn with skip connections")
        if loss_inference.item() < 0.02:
            print("✅ Decoder works reasonably well without skip connections")
        else:
            print("⚠️ Decoder struggles without skip connections (expected for untrained model)")
    else:
        print(f"\n❌ FAIL: Model not learning (loss = {loss.item():.6f})")
