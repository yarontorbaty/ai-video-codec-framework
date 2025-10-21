#!/usr/bin/env python3
"""
Full SOTA Training - Production Quality

Target: 28-32 dB PSNR, 0.92-0.95 SSIM
Training: 50 epochs, 50K samples
Expected Time: 8 hours on GPU
Expected Cost: ~$4
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

from models.enhanced_network import EnhancedPVCv2Model
from models.sota_residual_encoder import SOTAResidualEncoder
from models.sota_residual_decoder import SOTAResidualDecoder
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


def train_sota_full(
    num_samples=50000,
    num_epochs=50,
    batch_size=16,
    learning_rate=1e-4,
    quality_factor=30,
    save_path="/tmp",
    pvc_model_path="/tmp/pvc_v2_perceptual_best.pth",
    validate_every=5,
    save_every=10,
    checkpoint_path=None
):
    """
    Full SOTA training for production quality.
    
    Args:
        num_samples: Number of training samples (50K for full)
        num_epochs: Number of training epochs (50 for full)
        batch_size: Batch size (16 recommended for GPU)
        learning_rate: Learning rate (1e-4 is good)
        quality_factor: Quality factor for quantization (30 is good balance)
        save_path: Where to save models
        pvc_model_path: Path to pre-trained PVC model
        validate_every: Run validation every N epochs
        save_every: Save checkpoint every N epochs
        checkpoint_path: Path to resume from checkpoint (optional)
    
    Returns:
        dict: Training results
    """
    print("="*70)
    print("SOTA FULL TRAINING - Production Quality")
    print("="*70)
    print(f"\n⚙️  Configuration:")
    print(f"   Samples:       {num_samples:,}")
    print(f"   Epochs:        {num_epochs}")
    print(f"   Batch Size:    {batch_size}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Quality:       {quality_factor}")
    print(f"   Validation:    Every {validate_every} epochs")
    print(f"   Checkpoints:   Every {save_every} epochs")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📦 Device: {device}")
    
    if device.type == 'cpu':
        print("⚠️  WARNING: Running on CPU! This will be VERY slow.")
        print("   Estimated time: 80+ hours")
        print("   Recommendation: Use GPU worker instead")
        response = input("   Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Cancelled.")
            return None
    
    # Load pre-trained PVC model
    print(f"\n📦 Loading pre-trained PVC model...")
    pvc_model = EnhancedPVCv2Model(
        feature_dim=256,
        hidden_dim=128,
        num_functions=NUM_EXTENDED_FUNCTIONS + 1,
        max_sequence_length=20
    ).to(device)
    pvc_model.load_state_dict(torch.load(pvc_model_path, map_location=device))
    pvc_model.eval()
    print(f"✅ PVC model loaded")
    
    # Initialize SOTA models
    print(f"\n🏗️  Initializing SOTA models...")
    encoder = SOTAResidualEncoder(base_channels=64, quality_factor=quality_factor).to(device)
    decoder = SOTAResidualDecoder(base_channels=64).to(device)
    
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    total_params = encoder_params + decoder_params
    
    print(f"   Encoder: {encoder_params:,} params ({encoder_params*4/1024/1024:.1f} MB)")
    print(f"   Decoder: {decoder_params:,} params ({decoder_params*4/1024/1024:.1f} MB)")
    print(f"   Total:   {total_params:,} params ({total_params*4/1024/1024:.1f} MB)")
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"\n📂 Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"✅ Resumed from epoch {start_epoch}")
    
    # Optimizer
    model_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(model_params, lr=learning_rate, betas=(0.9, 0.999))
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Generate training data
    print(f"\n📊 Generating {num_samples:,} training samples...")
    print(f"   This will take ~{num_samples/1000:.1f} minutes...")
    
    generator = ExtendedSyntheticGenerator(width=256, height=256)
    
    start_gen = time.time()
    train_frames, _ = generator.generate_dataset(
        num_samples=num_samples,
        min_functions=5,
        max_functions=20
    )
    gen_time = time.time() - start_gen
    
    print(f"✅ Generated {num_samples:,} samples in {gen_time/60:.1f} minutes")
    print(f"   Frame shape: {train_frames.shape}")
    
    # Generate validation set
    print(f"\n📊 Generating 500 validation samples...")
    val_frames, _ = generator.generate_dataset(num_samples=500, min_functions=5, max_functions=20)
    print(f"✅ Generated validation set")
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {start_epoch} → {num_epochs}")
    print(f"   Expected time: ~{(num_samples/10000 * 95 * num_epochs)/60:.1f} hours")
    
    start_train = time.time()
    best_val_loss = float('inf')
    training_losses = []
    validation_losses = []
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        # Training
        encoder.train()
        decoder.train()
        
        epoch_loss = 0.0
        num_batches = len(train_frames) // batch_size
        
        indices = np.random.permutation(len(train_frames))
        
        for batch_idx in range(num_batches):
            batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_frames = train_frames[batch_indices]
            
            # Generate coarse reconstructions using PVC
            coarse_batch = []
            with torch.no_grad():
                for frame in batch_frames:
                    func_ids, params = pvc_model.predict_with_params(frame)
                    sparse_func_ids = [CONTIGUOUS_TO_SPARSE.get(fid, 0) for fid in func_ids]
                    
                    coarse = np.zeros_like(frame)
                    for fid, param in zip(sparse_func_ids, params):
                        coarse = execute_function_top10(fid, param, coarse)
                    coarse_batch.append(coarse)
            
            coarse_batch = np.array(coarse_batch)
            
            # Calculate residuals (ground truth)
            residuals_gt = batch_frames.astype(np.float32) - coarse_batch.astype(np.float32)
            residuals_gt = residuals_gt / 127.5  # Normalize to [-1, 1]
            
            # Convert to tensors
            residuals_tensor = torch.from_numpy(residuals_gt).permute(0, 3, 1, 2).float().to(device)
            
            # Forward pass
            optimizer.zero_grad()
            encoded = encoder(residuals_tensor)
            reconstructed = decoder(encoded)
            
            # Calculate loss
            loss = criterion(reconstructed, residuals_tensor)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_params, max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Progress update
            if (batch_idx + 1) % 50 == 0 or batch_idx == num_batches - 1:
                avg_loss = epoch_loss / (batch_idx + 1)
                progress = (batch_idx + 1) / num_batches * 100
                elapsed = time.time() - epoch_start
                eta_epoch = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                
                print(f"   Epoch {epoch+1}/{num_epochs} [{progress:5.1f}%] "
                      f"Loss: {avg_loss:.6f} "
                      f"ETA: {eta_epoch/60:.1f}m", end='\r')
        
        avg_epoch_loss = epoch_loss / num_batches
        training_losses.append(avg_epoch_loss)
        epoch_time = time.time() - epoch_start
        
        # Validation
        val_loss = None
        if (epoch + 1) % validate_every == 0:
            encoder.eval()
            decoder.eval()
            
            val_loss = 0.0
            val_batches = len(val_frames) // batch_size
            
            with torch.no_grad():
                for batch_idx in range(val_batches):
                    batch_frames = val_frames[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                    
                    # Generate coarse
                    coarse_batch = []
                    for frame in batch_frames:
                        func_ids, params = pvc_model.predict_with_params(frame)
                        sparse_func_ids = [CONTIGUOUS_TO_SPARSE.get(fid, 0) for fid in func_ids]
                        
                        coarse = np.zeros_like(frame)
                        for fid, param in zip(sparse_func_ids, params):
                            coarse = execute_function_top10(fid, param, coarse)
                        coarse_batch.append(coarse)
                    
                    coarse_batch = np.array(coarse_batch)
                    
                    # Calculate residuals
                    residuals_gt = batch_frames.astype(np.float32) - coarse_batch.astype(np.float32)
                    residuals_gt = residuals_gt / 127.5
                    
                    residuals_tensor = torch.from_numpy(residuals_gt).permute(0, 3, 1, 2).float().to(device)
                    
                    encoded = encoder(residuals_tensor)
                    reconstructed = decoder(encoded)
                    
                    loss = criterion(reconstructed, residuals_tensor)
                    val_loss += loss.item()
            
            val_loss /= val_batches
            validation_losses.append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(encoder.state_dict(), Path(save_path) / "sota_residual_encoder_best.pth")
                torch.save(decoder.state_dict(), Path(save_path) / "sota_residual_decoder_best.pth")
        
        # Print epoch summary
        elapsed_total = time.time() - start_train
        eta_total = elapsed_total / (epoch - start_epoch + 1) * (num_epochs - epoch - 1)
        eta_str = str(timedelta(seconds=int(eta_total)))
        
        print(f"   Epoch {epoch+1}/{num_epochs} "
              f"Train Loss: {avg_epoch_loss:.6f} "
              f"{f'Val Loss: {val_loss:.6f} ' if val_loss else ''}"
              f"Time: {epoch_time/60:.1f}m "
              f"ETA: {eta_str}     ")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': avg_epoch_loss,
                'val_loss': val_loss if val_loss else None,
                'best_val_loss': best_val_loss
            }
            checkpoint_path = Path(save_path) / f"sota_checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"   💾 Saved checkpoint: {checkpoint_path.name}")
    
    # Save final models
    print(f"\n💾 Saving final models...")
    torch.save(encoder.state_dict(), Path(save_path) / "sota_residual_encoder_final.pth")
    torch.save(decoder.state_dict(), Path(save_path) / "sota_residual_decoder_final.pth")
    
    total_time = time.time() - start_train
    
    print(f"\n" + "="*70)
    print(f"✅ TRAINING COMPLETE!")
    print(f"="*70)
    print(f"\n📊 Training Summary:")
    print(f"   Total Time:    {total_time/3600:.2f} hours")
    print(f"   Epochs:        {num_epochs}")
    print(f"   Final Loss:    {training_losses[-1]:.6f}")
    print(f"   Best Val Loss: {best_val_loss:.6f}")
    print(f"   Time/Epoch:    {total_time/num_epochs/60:.1f} minutes")
    
    if len(training_losses) > 1:
        improvement = (training_losses[0] - training_losses[-1]) / training_losses[0] * 100
        print(f"   Improvement:   {improvement:.1f}%")
    
    print(f"\n📦 Saved Models:")
    print(f"   Best:  sota_residual_encoder_best.pth")
    print(f"   Best:  sota_residual_decoder_best.pth")
    print(f"   Final: sota_residual_encoder_final.pth")
    print(f"   Final: sota_residual_decoder_final.pth")
    
    return {
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'best_val_loss': best_val_loss,
        'total_time': total_time,
        'num_epochs': num_epochs,
        'num_samples': num_samples
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Full SOTA Training')
    parser.add_argument('--samples', type=int, default=50000, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--quality', type=int, default=30, help='Quality factor')
    parser.add_argument('--save-path', type=str, default='/tmp', help='Save path')
    parser.add_argument('--pvc-model', type=str, default='/tmp/pvc_v2_perceptual_best.pth', help='PVC model path')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--validate-every', type=int, default=5, help='Validate every N epochs')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    results = train_sota_full(
        num_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        quality_factor=args.quality,
        save_path=args.save_path,
        pvc_model_path=args.pvc_model,
        checkpoint_path=args.checkpoint,
        validate_every=args.validate_every,
        save_every=args.save_every
    )
    
    if results:
        print(f"\n🎉 Training completed successfully!")
        print(f"   Ready for evaluation with eval_sota.py")

