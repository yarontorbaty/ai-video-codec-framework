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
import cv2
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.enhanced_network import EnhancedPVCv2Model
from training.perceptual_loss import CombinedLoss, VGGPerceptualLoss
from training.dataset_with_params import FunctionSequenceDatasetWithParams
from training.synthetic_generator_extended import ExtendedSyntheticGenerator
from graphics.primitives_extended import NUM_EXTENDED_FUNCTIONS, CONTIGUOUS_TO_SPARSE


def execute_function_top10(func_id: int, params: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """Execute top 10 graphics functions (simplified for training speed)."""
    height, width = canvas.shape[:2]
    
    # Denormalize parameters
    x1 = int(np.clip(params[0] * width, 0, width - 1))
    y1 = int(np.clip(params[1] * height, 0, height - 1))
    x2 = int(np.clip(params[2] * width, 0, width - 1))
    y2 = int(np.clip(params[3] * height, 0, height - 1))
    
    color1 = tuple(np.clip(params[4:7] * 255, 0, 255).astype(int).tolist())
    color2 = tuple(np.clip(params[7:10] * 255, 0, 255).astype(int).tolist())
    
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    
    try:
        if func_id == 0:  # fill_solid
            canvas[:] = color1
        elif func_id == 3:  # draw_rect
            if x2 > x1 and y2 > y1:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color1, -1)
        elif func_id == 5:  # draw_circle
            radius = max(5, min(w, h) // 2)
            cv2.circle(canvas, (cx, cy), radius, color1, -1)
        elif func_id == 10:  # fill_radial_gradient
            radius = max(10, (w + h) // 2)
            y_coords, x_coords = np.ogrid[:height, :width]
            dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            dist_norm = np.clip(dist / max(radius, 1), 0, 1)
            for c in range(3):
                canvas[:, :, c] = (color1[c] * (1 - dist_norm) + color2[c] * dist_norm).astype(np.uint8)
        elif func_id == 1 or func_id == 7:  # horizontal gradient
            for c in range(3):
                gradient = np.linspace(color1[c], color2[c], width)
                canvas[:, :, c] = np.tile(gradient, (height, 1)).astype(np.uint8)
        elif func_id == 6:  # vertical gradient
            for c in range(3):
                gradient = np.linspace(color1[c], color2[c], height)
                canvas[:, :, c] = np.tile(gradient.reshape(-1, 1), (1, width)).astype(np.uint8)
        elif func_id == 2:  # draw_ellipse
            rx = max(5, w // 2)
            ry = max(5, h // 2)
            cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, color1, -1)
        elif func_id == 13:  # fill_checkerboard
            square_size = max(8, min(32, w // 4))
            y_grid, x_grid = np.ogrid[:height, :width]
            pattern = ((x_grid // square_size) + (y_grid // square_size)) % 2
            for c in range(3):
                canvas[:, :, c] = np.where(pattern == 0, color1[c], color2[c]).astype(np.uint8)
        elif func_id == 23:  # draw_rounded_rect
            if x2 > x1 and y2 > y1:
                radius = min(10, w // 4, h // 4)
                cv2.rectangle(canvas, (x1 + radius, y1), (x2 - radius, y2), color1, -1)
                cv2.rectangle(canvas, (x1, y1 + radius), (x2, y2 - radius), color1, -1)
                if radius > 0:
                    cv2.circle(canvas, (x1 + radius, y1 + radius), radius, color1, -1)
                    cv2.circle(canvas, (x2 - radius, y1 + radius), radius, color1, -1)
                    cv2.circle(canvas, (x1 + radius, y2 - radius), radius, color1, -1)
                    cv2.circle(canvas, (x2 - radius, y2 - radius), radius, color1, -1)
        elif func_id == 24:  # draw_star
            outer_r = max(10, min(w, h) // 2)
            inner_r = outer_r // 2
            pts = []
            for i in range(10):
                angle = i * np.pi / 5 - np.pi / 2
                radius = outer_r if i % 2 == 0 else inner_r
                x = int(cx + radius * np.cos(angle))
                y = int(cy + radius * np.sin(angle))
                pts.append((x, y))
            pts_array = np.array(pts, dtype=np.int32)
            cv2.fillPoly(canvas, [pts_array], color1)
        else:
            # Fallback: average color
            avg_color = tuple(((np.array(color1) + np.array(color2)) / 2).astype(int).tolist())
            canvas[:] = avg_color
    except:
        pass
    
    return canvas


def reconstruct_batch(model, frames_batch, device):
    """
    Reconstruct a batch of frames using the model's predictions.
    
    Args:
        frames_batch: numpy array (B, H, W, 3) or torch tensor
        
    Returns reconstructed frames as tensors in [0, 1] range for perceptual loss.
    """
    # Convert to numpy if needed
    if isinstance(frames_batch, torch.Tensor):
        frames_batch = frames_batch.cpu().numpy()
    
    batch_size = frames_batch.shape[0]
    height, width = frames_batch.shape[1:3]
    
    reconstructed_batch = []
    
    model.eval()
    with torch.no_grad():
        for i in range(batch_size):
            frame = frames_batch[i]
            
            # Get predictions (in contiguous ID space)
            func_ids, params = model.predict_with_params(frame)
            
            # Convert contiguous IDs back to sparse IDs for execution
            sparse_func_ids = [CONTIGUOUS_TO_SPARSE.get(fid, 0) for fid in func_ids]
            
            # Reconstruct using top 10 functions (for speed)
            canvas = np.zeros_like(frame)
            for fid, param in zip(sparse_func_ids, params):
                canvas = execute_function_top10(fid, param, canvas)
            
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
        num_functions=NUM_EXTENDED_FUNCTIONS + 1,  # +1 for END token (43 total)
        max_sequence_length=20
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # Initialize combined loss (includes perceptual loss)
    print(f"\n🎯 Initializing combined loss...")
    print(f"   Weights: 0.3 function + 0.3 param + 0.4 perceptual")
    print(f"   Num classes: {NUM_EXTENDED_FUNCTIONS + 1} (including END token)")
    criterion = CombinedLoss(
        num_functions=NUM_EXTENDED_FUNCTIONS + 1,  # +1 for END token
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
    train_dataset = FunctionSequenceDatasetWithParams(
        train_frames,
        train_sequences,
        max_seq_len=20,
        end_token_id=NUM_EXTENDED_FUNCTIONS  # 42 for extended set
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
            
            # DEBUG: Check tensor values on first batch
            if batch_idx == 0 and epoch == 0:
                print(f"\n🔍 DEBUG - First Batch:")
                print(f"   func_ids_batch shape: {func_ids_batch.shape}")
                print(f"   func_ids_batch min: {func_ids_batch.min().item()}")
                print(f"   func_ids_batch max: {func_ids_batch.max().item()}")
                print(f"   Unique values: {torch.unique(func_ids_batch).cpu().tolist()}")
                print(f"   Model num_functions: {model.num_functions}")
                print(f"   Expected range: [0, {model.num_functions - 1}]")
                
                # Check if any values are out of range
                out_of_range = (func_ids_batch < 0) | (func_ids_batch >= model.num_functions)
                if out_of_range.any():
                    print(f"   ⚠️  WARNING: {out_of_range.sum().item()} values out of range!")
                    bad_values = func_ids_batch[out_of_range].unique().cpu().tolist()
                    print(f"   Bad values: {bad_values}")
            
            # Forward pass
            function_logits, predicted_sequences, predicted_params = model(frames_batch)
            
            # Reconstruct frames for perceptual loss (every 5 batches to save time)
            if batch_idx % 5 == 0:
                # Original frames are already in BCHW format from dataset, normalize to [0,1]
                original_frames = frames_batch.float()
                if original_frames.max() > 1.0:
                    original_frames = original_frames / 255.0
                
                # Convert to HWC format for reconstruction, then back to CHW
                frames_hwc = frames_batch.permute(0, 2, 3, 1)  # BCHW -> BHWC
                reconstructed_frames = reconstruct_batch(model, frames_hwc, device)
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

