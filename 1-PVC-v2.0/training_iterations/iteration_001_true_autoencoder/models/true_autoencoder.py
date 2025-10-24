#!/usr/bin/env python3
"""
True Autoencoder for Image Compression
No input needed at decode time - pure encode → latent → decode
"""

import torch
import torch.nn as nn


class CompressionAutoencoder(nn.Module):
    """
    True autoencoder for lossy image compression.
    
    Architecture:
    - Input: RGB image (3, H, W)
    - Encoder: Downsamples to compact latent (32, H/32, W/32)
    - Decoder: Reconstructs full image from latent alone
    - NO residual connection, NO input needed at decode time
    """
    
    def __init__(self, latent_channels=32):
        super().__init__()
        self.latent_channels = latent_channels
        
        # Encoder: 3 → 48 → 64 → 64 → 48 → 32 (5 downsampling stages = 32x compression)
        self.encoder = nn.Sequential(
            # Stage 1: 3 → 48, downsample 2x
            nn.Conv2d(3, 48, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(6, 48),
            nn.SiLU(),
            
            # Stage 2: 48 → 64, downsample 2x
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            # Stage 3: 64 → 64, downsample 2x
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            # Stage 4: 64 → 48, downsample 2x
            nn.Conv2d(64, 48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(6, 48),
            nn.SiLU(),
            
            # Stage 5: 48 → 32, downsample 2x
            nn.Conv2d(48, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, latent_channels),
            nn.SiLU(),
            
            # Final compression
            nn.Conv2d(latent_channels, latent_channels, kernel_size=1),
        )
        
        # Decoder: 32 → 48 → 64 → 64 → 48 → 3 (5 upsampling stages)
        self.decoder = nn.Sequential(
            # Initial processing
            nn.Conv2d(latent_channels, latent_channels, kernel_size=1),
            nn.SiLU(),
            
            # Stage 1: 32 → 48, upsample 2x
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(latent_channels, 48, kernel_size=3, padding=1),
            nn.GroupNorm(6, 48),
            nn.SiLU(),
            
            # Stage 2: 48 → 64, upsample 2x
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            # Stage 3: 64 → 64, upsample 2x
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            # Stage 4: 64 → 48, upsample 2x
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 48, kernel_size=3, padding=1),
            nn.GroupNorm(6, 48),
            nn.SiLU(),
            
            # Stage 5: 48 → 3, upsample 2x
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(48, 3, kernel_size=5, padding=2),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def encode(self, x):
        """Encode image to latent representation"""
        return self.encoder(x)
    
    def decode(self, latent):
        """Decode latent to image - NO input needed!"""
        return self.decoder(latent)
    
    def forward(self, x):
        """
        Full encode-decode cycle.
        Returns:
            output: Reconstructed image
            latent: Compressed representation
        """
        latent = self.encode(x)
        output = self.decode(latent)
        return output, latent


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for better visual quality"""
    
    def __init__(self):
        super().__init__()
        # Use VGG16 features
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)
        
        # Extract feature layers
        self.features = nn.Sequential(*list(vgg.features)[:16])  # Up to relu3_3
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.features.eval()
    
    def forward(self, pred, target):
        """Calculate perceptual loss"""
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        # Extract features
        pred_features = self.features(pred_norm)
        target_features = self.features(target_norm)
        
        # MSE on features
        return nn.functional.mse_loss(pred_features, target_features)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = CompressionAutoencoder(latent_channels=32)
    
    print("=== Compression Autoencoder ===")
    print(f"Total parameters: {count_parameters(model):,}")
    print()
    
    # Test with dummy input
    x = torch.randn(1, 3, 256, 256)
    
    print("Input shape:", x.shape)
    
    # Encode
    latent = model.encode(x)
    print("Latent shape:", latent.shape)
    print(f"Compression ratio: {x.numel() / latent.numel():.1f}x")
    
    # Decode
    output = model.decode(latent)
    print("Output shape:", output.shape)
    
    # Full forward
    output2, latent2 = model(x)
    print()
    print("✓ Model working correctly!")
    print(f"✓ Output range: [{output2.min():.3f}, {output2.max():.3f}]")

