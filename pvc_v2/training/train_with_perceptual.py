#!/usr/bin/env python3
"""
PVC v2.0 Training with Perceptual Loss

Trains the enhanced model with:
- Function prediction loss (CrossEntropy)
- Parameter prediction loss (MSE)
- Perceptual loss (VGG-based) - NEW!

Target: 15-18 dB PSNR
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.enhanced_network import EnhancedPVCv2Model
from training.perceptual_loss import CombinedLoss, VGGPerceptualLoss
from training.dataset_with_params import ExtendedFunctionDataset
from training.synthetic_generator_extended import ExtendedSyntheticGenerator
from graphics.primitives_extended import NUM_EXTENDED_FUNCTIONS
from tests.complete_evaluation import execute_function_complete


def reconstruct_batch(model, frames_batch, device):
    """
    Reconstruct a batch of frames using the model's predictions.
    
    Returns reconstructed frames as tensors in [0, 1] range for perceptual loss.
    """
    batch_size = frames_batch.shape[0]
    height, width = frames_batch.shape[1:3]
    
    reconstructed_batch = []
    
    model.eval()
    with torch.no_grad():
        for i in range(batch_size):
            frame = frames_batch[i].cpu().numpy()
            
            # Get predictions
            func_ids, params = model.predict_with_params(frame)
            
            # Reconstruct using complete function set
            canvas = np.zeros_like(frame)
            for fid, param in zip(func_ids, params):
                canvas = execute_function_complete(fid, param, canvas)
            
            # Convert to tensor [0, 1] range, shape (3, H, W)
            canvas_tensor = torch.from_numpy(canvas).float() / 255.0
            canvas_tensor = canvas_tensor.permute(2, 0, 1)  # HWC -> CHW
            reconstructed_batch.append(canvas_tensor)
    
    model.train()
    return torch.stack(reconstructed_batch).to(device)


def train_with_perceptual_loss(
    num_samples=10000,
    batch_size=16,
    num_epochs=30,
    learning_rate=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_path='/tmp',
    eval_every=5
):
    """
    Train PVC v2.0 model with perceptual loss.
    
    Args:
        num_samples: Number of training samples (10K for target PSNR)
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: torch device
        save_path: Where to save models
        eval_every: Evaluate reconstruction PSNR every N epochs
    """
    print("="*70)
    print("PVC v2.0 Training with Perceptual Loss")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    
    # Initialize model
    print(f"\n📦 Initializing model...")
    model = EnhancedPVCv2Model(
        feature_dim=256,
        hidden_dim=128,
        num_functions=NUM_EXTENDED_FUNCTIONS,
        max_sequence_length=20
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # Initialize combined loss (includes perceptual loss)
    print(f"\n🎯 Initializing combined loss...")
    print(f"   Weights: 0.3 function + 0.3 param + 0.4 perceptual")
    criterion = CombinedLoss(
        num_functions=NUM_EXTENDED_FUNCTIONS,
        device=device,
        weight_function=0.3,
        weight_param=0.3,
        weight_perceptual=0.4
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Generate training data
    print(f"\n📊 Generating {num_samples} training samples...")
    generator = ExtendedSyntheticGenerator(width=256, height=256)
    train_frames, train_sequences = generator.generate_dataset(
        num_samples=num_samples,
        min_functions=5,
        max_functions=20
    )
    
    # Create dataset and dataloader
    train_dataset = ExtendedFunctionDataset(
        train_frames,
        train_sequences,
        num_functions=NUM_EXTENDED_FUNCTIONS
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    print(f"   Batches per epoch: {len(train_loader)}")
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"="*70)
    
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {
            'total': [],
            'function': [],
            'param': [],
            'perceptual': []
        }
        
        for batch_idx, (frames_batch, func_ids_batch, params_batch) in enumerate(train_loader):
            frames_batch = frames_batch.to(device)
            func_ids_batch = func_ids_batch.to(device)
            params_batch = params_batch.to(device)
            
            # Forward pass
            function_logits, predicted_sequences, predicted_params = model(frames_batch)
            
            # Reconstruct frames for perceptual loss (every 5 batches to save time)
            if batch_idx % 5 == 0:
                # Convert original frames to [0, 1] range, CHW format
                original_frames = frames_batch.float() / 255.0
                original_frames = original_frames.permute(0, 3, 1, 2)  # BHWC -> BCHW
                
                # Reconstruct frames
                reconstructed_frames = reconstruct_batch(model, frames_batch.cpu().numpy(), device)
            else:
                original_frames = None
                reconstructed_frames = None
            
            # Compute combined loss
            loss, loss_dict = criterion(
                function_logits,
                predicted_params,
                predicted_sequences,
                func_ids_batch,
                params_batch,
                reconstructed_frames,
                original_frames
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Record losses
            for key in epoch_losses:
                epoch_losses[key].append(loss_dict[key])
        
        # Epoch summary
        avg_losses = {key: np.mean(vals) for key, vals in epoch_losses.items()}
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} ({elapsed/60:.1f}min) | "
              f"Loss: {avg_losses['total']:.4f} | "
              f"Func: {avg_losses['function']:.4f} | "
              f"Param: {avg_losses['param']:.4f} | "
              f"Perc: {avg_losses['perceptual']:.4f}")
        
        # Save best model
        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            torch.save(model.state_dict(), f"{save_path}/pvc_v2_perceptual_best.pth")
    
    # Save final model
    torch.save(model.state_dict(), f"{save_path}/pvc_v2_perceptual_final.pth")
    
    total_time = time.time() - start_time
    print(f"\n" + "="*70)
    print(f"✅ Training Complete!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Models saved: {save_path}/pvc_v2_perceptual_*.pth")
    print(f"="*70)
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PVC v2.0 with perceptual loss')
    parser.add_argument('--samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save-path', type=str, default='/tmp', help='Save path for models')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = train_with_perceptual_loss(
        num_samples=args.samples,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        save_path=args.save_path
    )
    
    print("\n🎉 Training finished! Run evaluation to measure PSNR.")

