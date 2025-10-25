#!/usr/bin/env python3
"""
Training script for Layer-Based Anime Codec

Trains the residual neural codec on real anime data (50K frames)
Uses existing anime_frames_960x540_50k.npy dataset on GPU worker
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.residual_codec import LayeredAnimeCodec
from utils.layer_extraction import extract_line_art, extract_color_palette, extract_residual, apply_palette

class AnimeResidualDataset(Dataset):
    """
    Dataset that extracts layers from real anime frames on-the-fly
    """
    def __init__(self, frames_npy_path, n_palette_colors=16):
        """
        Args:
            frames_npy_path: Path to .npy file with anime frames (N, H, W, 3) [0, 255]
            n_palette_colors: Number of colors in palette
        """
        print(f"Loading anime frames from {frames_npy_path}...")
        # Memory-map the file (don't load all into RAM)
        self.frames = np.load(frames_npy_path, mmap_mode='r')
        self.n_palette_colors = n_palette_colors
        print(f"✓ Loaded {len(self.frames)} frames")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        # Load frame [0, 255]
        frame = self.frames[idx].copy()  # Copy to make writable
        
        # Extract layers
        line_art = extract_line_art(frame, threshold1=50, threshold2=150)
        palette, color_map = extract_color_palette(frame, n_colors=self.n_palette_colors, mask=line_art)
        palette_frame = apply_palette(color_map, palette)
        residual = extract_residual(frame, line_art, palette_frame)
        
        # Convert to tensors [0, 1] or [-1, 1]
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # [0, 1]
        line_art_tensor = torch.from_numpy(line_art).unsqueeze(0).float() / 255.0  # [0, 1]
        palette_frame_tensor = torch.from_numpy(palette_frame).permute(2, 0, 1).float() / 255.0  # [0, 1]
        residual_tensor = torch.from_numpy(residual).permute(2, 0, 1).float() / 255.0  # Normalize to [-1, 1]
        residual_tensor = (residual_tensor - 0.5) * 2  # Scale to [-1, 1]
        
        return {
            'original': frame_tensor,
            'line_art': line_art_tensor,
            'palette': palette_frame_tensor,
            'residual': residual_tensor
        }


def train():
    print("="*70)
    print("TRAINING LAYER-BASED ANIME CODEC - RESIDUAL COMPONENT")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    # Hyperparameters
    latent_channels = 32
    n_palette_colors = 16
    batch_size = 8
    num_epochs = 100
    learning_rate = 1e-4
    
    print(f"\nHyperparameters:")
    print(f"  Latent channels: {latent_channels}")
    print(f"  Palette colors: {n_palette_colors}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    
    # Dataset (use existing 50K anime frames)
    dataset_path = '/home/ec2-user/pvc_phase25/anime_frames_960x540_50k.npy'
    dataset = AnimeResidualDataset(dataset_path, n_palette_colors=n_palette_colors)
    
    # Split into train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\n✓ Dataset loaded:")
    print(f"  Train: {len(train_dataset)} frames")
    print(f"  Val: {len(val_dataset)} frames")
    
    # Model
    model = LayeredAnimeCodec(latent_channels=latent_channels, n_palette_colors=n_palette_colors).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model: {total_params:,} parameters")
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    mse_loss = nn.MSELoss()
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"TRAINING START")
    print(f"{'='*70}\n")
    
    best_val_psnr = 0.0
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in train_loader:
            original = batch['original'].to(device)
            line_art = batch['line_art'].to(device)
            palette = batch['palette'].to(device)
            
            # Forward pass
            reconstructed, latent = model(original, line_art, palette)
            
            # Loss
            loss = mse_loss(reconstructed, original)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                original = batch['original'].to(device)
                line_art = batch['line_art'].to(device)
                palette = batch['palette'].to(device)
                
                reconstructed, latent = model(original, line_art, palette)
                loss = mse_loss(reconstructed, original)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        # Calculate PSNR
        train_psnr = -10 * np.log10(avg_train_loss)
        val_psnr = -10 * np.log10(avg_val_loss)
        
        # Print progress
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f}, PSNR: {train_psnr:.2f} dB")
        print(f"  Val Loss:   {avg_val_loss:.6f}, PSNR: {val_psnr:.2f} dB")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if device.type == 'cuda':
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  GPU Memory: {mem_allocated:.2f}GB / {mem_reserved:.2f}GB")
        print()
        
        # Save best model
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            save_path = Path(__file__).parent.parent / 'models' / 'layered_codec_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'val_loss': avg_val_loss,
            }, save_path)
            print(f"  ✓ Saved best model (PSNR: {val_psnr:.2f} dB)\n")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_path = Path(__file__).parent.parent / 'models' / f'layered_codec_epoch{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'val_loss': avg_val_loss,
            }, save_path)
        
        scheduler.step()
    
    print("="*70)
    print("✓ TRAINING COMPLETE!")
    print(f"✓ Best validation PSNR: {best_val_psnr:.2f} dB")
    print("="*70)


if __name__ == "__main__":
    train()

