#!/usr/bin/env python3
"""
Quick Test Training - Production Architecture Validation

Goal: Validate 93M param architecture can beat 25.06 dB baseline
Training: 100 epochs, 10K samples (reuse existing data)
Expected: 27-28 dB if architecture is good
Time: ~8-12 hours on g4dn.xlarge
Cost: ~$10
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.production_residual_encoder import ProductionResidualEncoder
from models.production_residual_decoder import ProductionResidualDecoder
from models.enhanced_network import EnhancedPVCv2Model
from training.synthetic_generator_extended import ExtendedSyntheticGenerator
from graphics.primitives_extended import NUM_EXTENDED_FUNCTIONS, CONTIGUOUS_TO_SPARSE


def execute_function_top10(func_id: int, params: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """Execute graphics functions for coarse reconstruction."""
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
        else:
            avg_color = tuple(((np.array(color1) + np.array(color2)) / 2).astype(int).tolist())
            canvas[:] = avg_color
    except:
        pass
    
    return canvas


def train_production_quick(
    num_samples=10000,
    num_epochs=100,
    batch_size=8,  # Smaller for 93M params
    learning_rate=1e-4,
    save_path="/tmp",
    pvc_model_path=None,
    validate_every=10,
    save_every=20
):
    """
    Quick test training for production architecture.
    
    Args:
        num_samples: 10K samples (reuse existing)
        num_epochs: 100 epochs (quick validation)
        batch_size: 8 (smaller for large model)
        learning_rate: 1e-4
        save_path: Where to save models
        pvc_model_path: Path to PVC model (if available)
        validate_every: Validate every N epochs
        save_every: Save checkpoint every N epochs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    if device.type == 'cpu':
        print("⚠️  WARNING: Training on CPU will be very slow!")
    
    # Initialize production models
    print("🏗️  Initializing production models...")
    encoder = ProductionResidualEncoder().to(device)
    decoder = ProductionResidualDecoder().to(device)
    
    enc_params = encoder.get_num_parameters()
    dec_params = decoder.get_num_parameters()
    total_params = enc_params + dec_params
    
    print(f"📊 Model Parameters:")
    print(f"   Encoder: {enc_params / 1e6:.2f}M")
    print(f"   Decoder: {dec_params / 1e6:.2f}M")
    print(f"   Total: {total_params / 1e6:.2f}M (vs 32.4M baseline)")
    
    # Load PVC model for coarse reconstruction (if available)
    pvc_model = None
    if pvc_model_path and Path(pvc_model_path).exists():
        print(f"📂 Loading PVC model from {pvc_model_path}")
        try:
            pvc_model = EnhancedPVCv2Model(
                feature_dim=256,
                hidden_dim=128,
                num_functions=NUM_EXTENDED_FUNCTIONS + 1,
                max_sequence_length=20
            ).to(device)
            pvc_model.load_state_dict(torch.load(pvc_model_path, map_location=device))
            pvc_model.eval()
            print("✅ PVC model loaded")
        except Exception as e:
            print(f"⚠️  Could not load PVC model: {e}")
            pvc_model = None
    
    # Optimizer with gradient accumulation
    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # Learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Generate training data
    print(f"\n📊 Generating {num_samples} training samples...")
    generator = ExtendedSyntheticGenerator(img_size=256)
    
    frames = []
    start_gen = time.time()
    for i in range(num_samples):
        frame = generator.generate_frame()
        frames.append(frame)
        
        if (i + 1) % 1000 == 0:
            print(f"   Generated {i + 1}/{num_samples} samples...")
    
    frames = np.array(frames)
    gen_time = time.time() - start_gen
    print(f"✅ Data generation complete in {gen_time / 60:.1f} minutes")
    print(f"   Data shape: {frames.shape}")
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Samples per epoch: {num_samples}")
    print(f"   Learning rate: {learning_rate}")
    
    best_loss = float('inf')
    training_start = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        encoder.train()
        decoder.train()
        
        # Shuffle data
        indices = np.random.permutation(num_samples)
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx in range(0, num_samples, batch_size):
            batch_indices = indices[batch_idx:batch_idx + batch_size]
            batch_frames = frames[batch_indices]
            
            # Convert to tensor
            batch_frames_tensor = torch.from_numpy(batch_frames).float().permute(0, 3, 1, 2).to(device)
            batch_frames_tensor = (batch_frames_tensor / 255.0) * 2 - 1  # Normalize to [-1, 1]
            
            # Generate coarse reconstruction with PVC (if available)
            coarse_recons = []
            for frame in batch_frames:
                canvas = np.zeros_like(frame)
                
                if pvc_model is not None:
                    # Use PVC model for coarse reconstruction
                    with torch.no_grad():
                        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0).to(device)
                        frame_tensor = (frame_tensor / 255.0) * 2 - 1
                        func_ids, params = pvc_model.predict_with_params(frame_tensor)
                        
                        # Convert to sparse IDs and execute
                        for fid, param in zip(func_ids[:10], params[:10]):
                            sparse_fid = CONTIGUOUS_TO_SPARSE.get(fid, 0)
                            canvas = execute_function_top10(sparse_fid, param, canvas)
                else:
                    # Simple average color if no PVC
                    canvas[:] = frame.mean(axis=(0, 1))
                
                coarse_recons.append(canvas)
            
            coarse_recons = np.array(coarse_recons)
            coarse_tensor = torch.from_numpy(coarse_recons).float().permute(0, 3, 1, 2).to(device)
            coarse_tensor = (coarse_tensor / 255.0) * 2 - 1
            
            # Calculate residual
            residual = batch_frames_tensor - coarse_tensor
            
            # Forward pass
            latent, skips = encoder(residual)
            recon_residual = decoder(latent, skips)
            
            # Loss
            loss = criterion(recon_residual, residual)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                max_norm=1.0
            )
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average loss
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start
        
        # Estimate remaining time
        elapsed = time.time() - training_start
        epochs_done = epoch + 1
        epochs_remaining = num_epochs - epochs_done
        time_per_epoch = elapsed / epochs_done
        eta = time_per_epoch * epochs_remaining
        eta_str = str(timedelta(seconds=int(eta)))
        
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {epoch_time:.1f}s | ETA: {eta_str}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), Path(save_path) / "production_encoder_best.pth")
            torch.save(decoder.state_dict(), Path(save_path) / "production_decoder_best.pth")
            print(f"   ✅ Saved best model (loss: {best_loss:.6f})")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save(encoder.state_dict(), Path(save_path) / f"production_encoder_epoch{epoch + 1}.pth")
            torch.save(decoder.state_dict(), Path(save_path) / f"production_decoder_epoch{epoch + 1}.pth")
            print(f"   💾 Saved checkpoint at epoch {epoch + 1}")
        
        # Validation
        if (epoch + 1) % validate_every == 0:
            encoder.eval()
            decoder.eval()
            
            with torch.no_grad():
                # Sample validation batch
                val_indices = np.random.choice(num_samples, min(100, num_samples), replace=False)
                val_frames = frames[val_indices]
                
                psnrs = []
                for frame in val_frames:
                    # Original
                    frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0).to(device)
                    frame_tensor = (frame_tensor / 255.0) * 2 - 1
                    
                    # Coarse reconstruction
                    canvas = np.zeros_like(frame)
                    if pvc_model is not None:
                        with torch.no_grad():
                            func_ids, params = pvc_model.predict_with_params(frame_tensor)
                            for fid, param in zip(func_ids[:10], params[:10]):
                                sparse_fid = CONTIGUOUS_TO_SPARSE.get(fid, 0)
                                canvas = execute_function_top10(sparse_fid, param, canvas)
                    else:
                        canvas[:] = frame.mean(axis=(0, 1))
                    
                    coarse_tensor = torch.from_numpy(canvas).float().permute(2, 0, 1).unsqueeze(0).to(device)
                    coarse_tensor = (coarse_tensor / 255.0) * 2 - 1
                    
                    # Residual
                    residual = frame_tensor - coarse_tensor
                    
                    # Encode-decode
                    latent, skips = encoder(residual)
                    recon_residual = decoder(latent, skips)
                    
                    # Final reconstruction
                    final_recon = coarse_tensor + recon_residual
                    
                    # Calculate PSNR
                    mse = torch.mean((frame_tensor - final_recon) ** 2).item()
                    if mse > 0:
                        psnr = 10 * np.log10(4.0 / mse)  # Range is [-1, 1], so max diff is 2, squared is 4
                        psnrs.append(psnr)
                
                avg_psnr = np.mean(psnrs) if psnrs else 0
                print(f"   📊 Validation PSNR: {avg_psnr:.2f} dB (target: 27-28 dB)")
    
    total_time = time.time() - training_start
    print(f"\n✅ Training complete!")
    print(f"   Total time: {total_time / 3600:.2f} hours")
    print(f"   Best loss: {best_loss:.6f}")
    print(f"   Final learning rate: {scheduler.get_last_lr()[0]:.2e}")
    
    return encoder, decoder


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick test training for production architecture")
    parser.add_argument("--samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="/tmp", help="Save path")
    parser.add_argument("--pvc-model", type=str, default=None, help="Path to PVC model")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PVC v2.0 Production Architecture - Quick Test Training")
    print("=" * 80)
    print(f"Target: Validate 93M param architecture (baseline: 25.06 dB)")
    print(f"Expected: 27-28 dB if architecture is good")
    print(f"Time: 8-12 hours on GPU")
    print("=" * 80)
    print()
    
    encoder, decoder = train_production_quick(
        num_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path,
        pvc_model_path=args.pvc_model
    )
    
    print("\n🎉 Quick test complete! Check models at:", args.save_path)

