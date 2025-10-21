#!/usr/bin/env python3
"""
Multi-GPU Production Training - Optimized for 4× V100 GPUs

Fast training using DataParallel for the production architecture.
Target: Complete 100 epochs in ~8-10 minutes on p3.8xlarge.
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
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.production_residual_encoder import ProductionResidualEncoder
from models.production_residual_decoder import ProductionResidualDecoder
from training.synthetic_generator_extended import ExtendedSyntheticGenerator


def train_production_multigpu(
    num_samples=10000,
    num_epochs=100,
    batch_size=128,  # Optimized for 4 GPUs (128 per GPU = 512 effective)
    learning_rate=1e-4,
    save_path="/tmp/models",
    save_every=10,
    validate_every=10
):
    """
    Multi-GPU training for production architecture.
    """
    
    # Check for multiple GPUs
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    num_gpus = torch.cuda.device_count()
    print(f"""
================================================================================
PVC v2.0 Production Architecture - Multi-GPU Training
================================================================================
Target: Validate 93M param architecture (baseline: 25.06 dB)
Expected: 27-28 dB if architecture is good
GPUs: {num_gpus}× {torch.cuda.get_device_name(0)}
Time: ~8-10 minutes on 4× V100
================================================================================
""")
    
    device = torch.device("cuda")
    
    # Initialize models
    print("🏗️  Initializing production models...")
    encoder = ProductionResidualEncoder().to(device)
    decoder = ProductionResidualDecoder().to(device)
    
    # Wrap with DataParallel for multi-GPU
    if num_gpus > 1:
        print(f"📊 Using DataParallel across {num_gpus} GPUs")
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
    
    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"📊 Model Parameters:")
    print(f"   Encoder: {encoder_params / 1e6:.2f}M")
    print(f"   Decoder: {decoder_params / 1e6:.2f}M")
    print(f"   Total: {(encoder_params + decoder_params) / 1e6:.2f}M (vs 32.4M baseline)")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Generate training data
    print(f"\n📊 Generating {num_samples} training samples...")
    generator = ExtendedSyntheticGenerator(width=256, height=256)
    
    frames = []
    start_gen = time.time()
    for i in range(num_samples):
        frame, _ = generator.generate_scene(num_functions=10)
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
    print(f"   Batch size: {batch_size} (×{num_gpus} GPUs = {batch_size * num_gpus} effective)")
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
        
        # Mini-batch training
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_frames = frames[batch_indices]
            
            # Convert to tensor
            batch_frames_tensor = torch.from_numpy(batch_frames).float().permute(0, 3, 1, 2).to(device)
            batch_frames_tensor = (batch_frames_tensor / 255.0) * 2 - 1  # Normalize to [-1, 1]
            
            # Simple coarse reconstruction (average color)
            coarse_recons = []
            for frame in batch_frames:
                canvas = np.zeros_like(frame)
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
              f"Time: {epoch_time:.1f}s | ETA: {eta_str}", flush=True)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Handle DataParallel wrapper
            encoder_state = encoder.module.state_dict() if hasattr(encoder, 'module') else encoder.state_dict()
            decoder_state = decoder.module.state_dict() if hasattr(decoder, 'module') else decoder.state_dict()
            
            torch.save(encoder_state, Path(save_path) / "production_encoder_best.pth")
            torch.save(decoder_state, Path(save_path) / "production_decoder_best.pth")
            print(f"   ✅ Saved best model (loss: {best_loss:.6f})", flush=True)
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            encoder_state = encoder.module.state_dict() if hasattr(encoder, 'module') else encoder.state_dict()
            decoder_state = decoder.module.state_dict() if hasattr(decoder, 'module') else decoder.state_dict()
            
            torch.save(encoder_state, Path(save_path) / f"production_encoder_epoch{epoch + 1}.pth")
            torch.save(decoder_state, Path(save_path) / f"production_decoder_epoch{epoch + 1}.pth")
            print(f"   💾 Saved checkpoint at epoch {epoch + 1}", flush=True)
        
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
                    
                    # Coarse reconstruction (average color)
                    canvas = np.zeros_like(frame)
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
                print(f"   📊 Validation PSNR: {avg_psnr:.2f} dB (target: 27-28 dB)", flush=True)
    
    total_time = time.time() - training_start
    
    print(f"""
================================================================================
✅ Training Complete!
================================================================================
Total time: {total_time / 60:.1f} minutes
Best loss: {best_loss:.6f}
Models saved to: {save_path}
================================================================================
""")
    
    return encoder, decoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--save-path", type=str, default="/home/ec2-user/pvc_training/models")
    args = parser.parse_args()
    
    # Create save directory
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    
    train_production_multigpu(
        num_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_path
    )

