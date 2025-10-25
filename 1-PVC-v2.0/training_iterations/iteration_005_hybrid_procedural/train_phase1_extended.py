#!/usr/bin/env python3
"""
Phase 1 Extended Training: Train Procedural Predictor for 200 epochs

Goal: Max out procedural predictor quality before adding residual
Target: 15-18 dB PSNR with procedural operations alone
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from hybrid_model import ProceduralPredictor
from renderer import render_anime_frame

class SimpleAnimeDataset(Dataset):
    """Simple dataset for quick testing"""
    def __init__(self, num_samples=1000, size=(512, 960)):
        self.num_samples = num_samples
        self.height, self.width = size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic anime-style frame
        np.random.seed(idx)
        
        # Random operations
        num_ops = np.random.randint(5, 15)
        operations = []
        for _ in range(num_ops):
            func_id = np.random.randint(0, 15)
            params = np.random.rand(10).astype(np.float32)
            operations.append((func_id, params))
        
        # Render frame
        frame = render_anime_frame(operations, self.width, self.height)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        # Return ONLY the frame tensor (not operations - they're different lengths!)
        return frame_tensor


class DifferentiableRenderer(nn.Module):
    """Wrapper to make anime renderer work in training loop"""
    def __init__(self, width=960, height=512):
        super().__init__()
        self.width = width
        self.height = height
    
    def forward(self, function_ids, params):
        """
        Args:
            function_ids: (B, max_ops) tensor of function IDs
            params: (B, max_ops, 10) tensor of parameters
        Returns:
            rendered: (B, 3, H, W) tensor [0, 1]
        """
        batch_size = function_ids.size(0)
        rendered_frames = []
        
        for b in range(batch_size):
            # Convert to operations list
            operations = []
            for i in range(function_ids.size(1)):
                func_id = function_ids[b, i].item()
                if func_id == 15:  # END token
                    break
                func_params = params[b, i].detach().cpu().numpy()
                operations.append((func_id, func_params))
            
            # Render frame
            if len(operations) > 0:
                frame = render_anime_frame(operations, self.width, self.height)
            else:
                frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
            
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            rendered_frames.append(frame_tensor)
        
        return torch.stack(rendered_frames).to(function_ids.device)


def train_phase1_extended():
    print("="*70)
    print("PHASE 1 EXTENDED: Training Procedural Predictor for 200 Epochs")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    # Initialize model from scratch
    model = ProceduralPredictor(num_functions=15, max_operations=50).to(device)
    print(f"✓ Initialized fresh ProceduralPredictor model")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {total_params:,}")
    
    # Initialize renderer
    renderer = DifferentiableRenderer(width=960, height=512).to(device)
    
    # Dataset and dataloader
    dataset = SimpleAnimeDataset(num_samples=5000, size=(512, 960))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    print(f"✓ Dataset: {len(dataset)} samples, batch_size=4")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 200
    
    print(f"\n{'='*70}")
    print(f"Training from Epoch 1 to {num_epochs}")
    print(f"{'='*70}\n")
    
    best_psnr = 0.0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_sparsity = 0
        num_batches = 0
        
        for batch_idx, img in enumerate(dataloader):
            img = img.to(device)
            
            # Forward pass
            function_logits, params = model(img)
            
            # Render procedural base
            with torch.no_grad():
                procedural_base = renderer(torch.argmax(function_logits, dim=-1), params)
            
            # Reconstruction loss
            recon_loss = mse_loss(procedural_base, img)
            
            # Sparsity loss (encourage END tokens)
            end_token_id = 15
            sparsity_loss = -torch.mean(function_logits[:, :, end_token_id])
            
            # Total loss
            loss = recon_loss + 0.01 * sparsity_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Stats
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_sparsity += sparsity_loss.item()
            num_batches += 1
        
        # Calculate PSNR
        avg_loss = total_loss / num_batches
        avg_recon = total_recon_loss / num_batches
        avg_sparsity = total_sparsity / num_batches
        psnr = -10 * np.log10(avg_recon)
        
        # Print progress
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  Loss: {avg_loss:.6f}")
        print(f"  Reconstruction: {avg_recon:.6f}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  Sparsity: {avg_sparsity:.6f}")
        
        if device.type == 'mps':
            # MPS doesn't have memory stats, skip
            pass
        elif device.type == 'cuda':
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        print()
        
        # Save best model
        if psnr > best_psnr:
            best_psnr = psnr
            save_path = Path(__file__).parent / "hybrid_phase1_best.pth"
            torch.save({
                'epoch': epoch,
                'procedural_predictor': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'psnr': psnr,
                'loss': avg_loss,
            }, save_path)
        
        # Save checkpoint every 50 epochs
        if epoch % 50 == 0:
            save_path = Path(__file__).parent / f"hybrid_phase1_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'procedural_predictor': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'psnr': psnr,
                'loss': avg_loss,
            }, save_path)
            print(f"✓ Saved checkpoint: {save_path.name}")
            print()
    
    print("="*70)
    print("✓ Phase 1 Extended Training Complete!")
    print(f"✓ Best PSNR: {best_psnr:.2f} dB")
    print(f"✓ Saved: hybrid_phase1_best.pth")
    print("="*70)


if __name__ == "__main__":
    train_phase1_extended()

