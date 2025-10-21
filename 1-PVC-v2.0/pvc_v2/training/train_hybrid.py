#!/usr/bin/env python3
"""
Train Hybrid PVC v2.0 - Residual Module Only

Strategy:
1. Load pre-trained PVC v2.0 model (frozen)
2. Train only the residual encoder/decoder
3. Loss: MSE on residuals + size penalty
4. Target: 30-40 dB final PSNR

Training time: 2-3 hours on GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.enhanced_network import EnhancedPVCv2Model
from models.residual_encoder import ResidualEncoder
from models.residual_decoder import ResidualDecoder
from training.synthetic_generator_extended import ExtendedSyntheticGenerator
from graphics.primitives_extended import NUM_EXTENDED_FUNCTIONS, CONTIGUOUS_TO_SPARSE
from skimage.metrics import peak_signal_noise_ratio
import cv2


class ResidualDataset(Dataset):
    """Dataset of (original, coarse, residual) triplets."""
    
    def __init__(self, frames, coarse_frames):
        self.frames = frames
        self.coarse_frames = coarse_frames
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        original = self.frames[idx]
        coarse = self.coarse_frames[idx]
        
        # Compute residual
        residual = original.astype(np.float32) - coarse.astype(np.float32)
        residual = residual / 127.5  # Normalize to [-2, 2]
        
        # Convert to tensors (C, H, W)
        original_tensor = torch.from_numpy(original).permute(2, 0, 1).float() / 255.0
        coarse_tensor = torch.from_numpy(coarse).permute(2, 0, 1).float() / 255.0
        residual_tensor = torch.from_numpy(residual).permute(2, 0, 1).float()
        
        return original_tensor, coarse_tensor, residual_tensor


def execute_function_top10(func_id: int, params: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """Execute graphics functions."""
    height, width = canvas.shape[:2]
    
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
        if func_id == 0:
            canvas[:] = color1
        elif func_id == 3:
            if x2 > x1 and y2 > y1:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color1, -1)
        elif func_id == 5:
            radius = max(5, min(w, h) // 2)
            cv2.circle(canvas, (cx, cy), radius, color1, -1)
        elif func_id == 10:
            radius = max(10, (w + h) // 2)
            y_coords, x_coords = np.ogrid[:height, :width]
            dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            dist_norm = np.clip(dist / max(radius, 1), 0, 1)
            for c in range(3):
                canvas[:, :, c] = (color1[c] * (1 - dist_norm) + color2[c] * dist_norm).astype(np.uint8)
        elif func_id == 1 or func_id == 7:
            for c in range(3):
                gradient = np.linspace(color1[c], color2[c], width)
                canvas[:, :, c] = np.tile(gradient, (height, 1)).astype(np.uint8)
        else:
            avg_color = tuple(((np.array(color1) + np.array(color2)) / 2).astype(int).tolist())
            canvas[:] = avg_color
    except:
        pass
    
    return canvas


def generate_coarse_reconstructions(pvc_model, frames, device):
    """Generate coarse reconstructions using PVC model."""
    coarse_frames = []
    
    pvc_model.eval()
    with torch.no_grad():
        for i, frame in enumerate(frames):
            if (i + 1) % 500 == 0:
                print(f"   Generated {i + 1}/{len(frames)} coarse frames...")
            
            func_ids, params = pvc_model.predict_with_params(frame)
            sparse_func_ids = [CONTIGUOUS_TO_SPARSE.get(fid, 0) for fid in func_ids]
            
            canvas = np.zeros_like(frame)
            for fid, param in zip(sparse_func_ids, params):
                canvas = execute_function_top10(fid, param, canvas)
            
            coarse_frames.append(canvas)
    
    return np.array(coarse_frames)


def train_hybrid_residual(
    num_samples=10000,
    num_epochs=20,
    batch_size=16,
    learning_rate=1e-4,
    quality_factor=20,
    save_path="/tmp",
    pvc_model_path="/tmp/pvc_v2_perceptual_best.pth"
):
    """Train residual encoder/decoder for hybrid codec."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("Hybrid PVC v2.0 - Residual Training")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Samples: {num_samples}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Quality factor: {quality_factor}")
    print(f"  Device: {device}")
    
    # Load PVC model (frozen)
    print(f"\n📦 Loading PVC v2.0 model...")
    pvc_model = EnhancedPVCv2Model(
        feature_dim=256,
        hidden_dim=128,
        num_functions=NUM_EXTENDED_FUNCTIONS + 1,
        max_sequence_length=20
    ).to(device)
    
    pvc_model.load_state_dict(torch.load(pvc_model_path, map_location=device))
    pvc_model.eval()
    for param in pvc_model.parameters():
        param.requires_grad = False
    print("✅ PVC model loaded and frozen")
    
    # Initialize residual codec
    print(f"\n📦 Initializing residual codec...")
    residual_encoder = ResidualEncoder(quality_factor=quality_factor).to(device)
    residual_decoder = ResidualDecoder().to(device)
    print("✅ Residual codec initialized")
    
    # Generate training data
    print(f"\n📊 Generating {num_samples} training samples...")
    generator = ExtendedSyntheticGenerator(width=256, height=256)
    frames, _ = generator.generate_dataset(
        num_samples=num_samples,
        min_functions=5,
        max_functions=20
    )
    print(f"✅ Generated {num_samples} original frames")
    
    # Generate coarse reconstructions
    print(f"\n🎨 Generating coarse reconstructions...")
    coarse_frames = generate_coarse_reconstructions(pvc_model, frames, device)
    print(f"✅ Generated {num_samples} coarse frames")
    
    # Calculate baseline coarse PSNR
    coarse_psnrs = [peak_signal_noise_ratio(frames[i], coarse_frames[i], data_range=255) 
                    for i in range(min(100, len(frames)))]
    avg_coarse_psnr = np.mean(coarse_psnrs)
    print(f"   Baseline coarse PSNR: {avg_coarse_psnr:.2f} dB")
    
    # Create dataset
    train_dataset = ResidualDataset(frames, coarse_frames)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    print(f"   Batches per epoch: {len(train_loader)}")
    
    # Optimizer and loss
    params = list(residual_encoder.parameters()) + list(residual_decoder.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)
    mse_criterion = nn.MSELoss()
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print("="*70)
    
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        residual_encoder.train()
        residual_decoder.train()
        
        epoch_losses = []
        
        for batch_idx, (original, coarse, residual_gt) in enumerate(train_loader):
            original = original.to(device)
            coarse = coarse.to(device)
            residual_gt = residual_gt.to(device)
            
            # Encode residuals
            compressed = residual_encoder(residual_gt)
            
            # Decode residuals
            residual_pred = residual_decoder(compressed)
            
            # Loss: MSE on residuals
            loss = mse_criterion(residual_pred, residual_gt)
            
            # Add size penalty (encourage sparsity)
            size_penalty = torch.mean(torch.abs(compressed['quantized_dct'])) * 0.001
            total_loss = loss + size_penalty
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Calculate epoch stats
        avg_loss = np.mean(epoch_losses)
        elapsed = (time.time() - start_time) / 60
        
        print(f"Epoch {epoch+1}/{num_epochs} ({elapsed:.1f}min) | Loss: {avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(residual_encoder.state_dict(), f"{save_path}/hybrid_residual_encoder_best.pth")
            torch.save(residual_decoder.state_dict(), f"{save_path}/hybrid_residual_decoder_best.pth")
    
    # Save final models
    torch.save(residual_encoder.state_dict(), f"{save_path}/hybrid_residual_encoder_final.pth")
    torch.save(residual_decoder.state_dict(), f"{save_path}/hybrid_residual_decoder_final.pth")
    
    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*70}")
    print(f"✅ Training Complete!")
    print(f"   Total time: {elapsed:.1f} minutes")
    print(f"   Best loss: {best_loss:.6f}")
    print(f"   Models saved: {save_path}/hybrid_residual_*.pth")
    print(f"{'='*70}")
    
    return residual_encoder, residual_decoder


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Hybrid PVC v2.0 Residual Codec")
    parser.add_argument("--samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--quality", type=int, default=20, help="Quality factor (10-50)")
    parser.add_argument("--save-path", type=str, default="/home/ec2-user", help="Path to save models")
    parser.add_argument("--pvc-model", type=str, default="/home/ec2-user/pvc_v2_perceptual_best.pth", 
                        help="Path to PVC model")
    
    args = parser.parse_args()
    
    encoder, decoder = train_hybrid_residual(
        num_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        quality_factor=args.quality,
        save_path=args.save_path,
        pvc_model_path=args.pvc_model
    )

