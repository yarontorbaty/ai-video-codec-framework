#!/usr/bin/env python3
"""
PVC v2.0 Training with Parameter Supervision

This is the breakthrough: Add parameter loss to achieve visual quality!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
from pathlib import Path
import time
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.network import PVCv2Model
from models.reconstructor import PVCv2Reconstructor
from training.synthetic_generator import SyntheticDataGenerator
from training.dataset_with_params import FunctionSequenceDatasetWithParams


def train_with_parameter_supervision(num_samples: int = 5000, num_epochs: int = 30):
    """
    Train PVC v2.0 with parameter supervision for visual quality.
    
    KEY INNOVATION: Add parameter loss alongside function ID loss!
    
    Args:
        num_samples: Number of training samples
        num_epochs: Number of training epochs
    """
    print("="*70)
    print("PVC v2.0 - Training with Parameter Supervision")
    print("="*70)
    print("\n🔑 KEY INNOVATION: Parameter Loss for Visual Quality!")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    print()
    
    # Generate training data
    print(f"📊 Generating {num_samples} training samples...")
    generator = SyntheticDataGenerator(width=256, height=256)
    frames, function_sequences = generator.generate_dataset(
        num_samples=num_samples,
        anime_ratio=0.6  # More anime-like content
    )
    print()
    
    # Create enhanced dataset (with parameters!)
    print("📦 Creating enhanced dataset with parameter ground truth...")
    dataset = FunctionSequenceDatasetWithParams(frames, function_sequences)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    print(f"   Dataset size: {len(dataset)} samples")
    print(f"   Batch size: 16")
    print()
    
    # Create model
    print("🧠 Creating neural network...")
    model = PVCv2Model(
        feature_dim=256,
        hidden_dim=256,
        num_functions=10,
        max_sequence_length=20
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    print()
    
    # Setup training with BOTH losses!
    print("⚙️  Setting up training with dual loss...")
    func_criterion = nn.CrossEntropyLoss()
    param_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print(f"   Optimizer: Adam (lr=0.001, decay every 10 epochs)")
    print(f"   Loss 1: CrossEntropyLoss (function IDs)")
    print(f"   Loss 2: MSELoss (parameters) ← NEW!")
    print(f"   Combined: total_loss = func_loss + 0.5 * param_loss")
    print()
    
    # Train
    print(f"🚀 Training for {num_epochs} epochs...")
    print()
    
    model.train()
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_func_loss = 0.0
        epoch_param_loss = 0.0
        epoch_total_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        for batch_idx, (frames_batch, func_ids_batch, params_batch) in enumerate(dataloader):
            frames_batch = frames_batch.to(device)
            func_ids_batch = func_ids_batch.to(device)
            params_batch = params_batch.to(device)  # (batch, seq_len, 10)
            
            optimizer.zero_grad()
            
            # Forward pass
            function_logits, predicted_sequences, _ = model(frames_batch, func_ids_batch)
            
            # Loss 1: Function ID prediction
            seq_len = min(function_logits.size(1), func_ids_batch.size(1))
            func_loss = 0
            for t in range(seq_len):
                func_loss += func_criterion(function_logits[:, t, :], func_ids_batch[:, t])
            func_loss = func_loss / seq_len
            
            # Loss 2: Parameter prediction (NEW!)
            param_loss = 0
            features = model.encoder(frames_batch)  # Get features
            
            for t in range(min(seq_len, params_batch.size(1))):
                # Get ground truth function ID
                true_func_id = func_ids_batch[:, t]
                
                # Skip END tokens (10)
                mask = (true_func_id != 10).float()
                if mask.sum() == 0:
                    continue
                
                # Predict parameters for this function
                predicted_params_dict = model.param_predictor(features, true_func_id)
                
                # Ground truth parameters
                true_params = params_batch[:, t]  # (batch, 10)
                true_coords = true_params[:, :4]
                true_color1 = true_params[:, 4:7]
                true_color2 = true_params[:, 7:10]
                
                # Compute MSE for each parameter type
                # Only for non-END tokens
                coord_loss = param_criterion(predicted_params_dict['coords'] * mask.unsqueeze(1), 
                                             true_coords * mask.unsqueeze(1))
                color1_loss = param_criterion(predicted_params_dict['color1'] * mask.unsqueeze(1),
                                             true_color1 * mask.unsqueeze(1))
                color2_loss = param_criterion(predicted_params_dict['color2'] * mask.unsqueeze(1),
                                             true_color2 * mask.unsqueeze(1))
                
                param_loss += (coord_loss + color1_loss + color2_loss) / 3.0
            
            if seq_len > 0:
                param_loss = param_loss / seq_len
            
            # Total loss: Combine function + parameter losses
            total_loss = func_loss + 0.5 * param_loss
            
            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            # Accuracy
            predicted = torch.argmax(function_logits[:, :seq_len, :], dim=2)
            target = func_ids_batch[:, :seq_len]
            mask = (target != 10).float()
            if mask.sum() > 0:
                correct = ((predicted == target).float() * mask).sum() / mask.sum()
            else:
                correct = torch.tensor(0.0)
            
            epoch_func_loss += func_loss.item()
            epoch_param_loss += param_loss.item()
            epoch_total_loss += total_loss.item()
            epoch_acc += correct.item()
            num_batches += 1
            
            # Print progress
            if (batch_idx + 1) % 30 == 0:
                print(f"   Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(dataloader)} - "
                      f"Loss: {total_loss.item():.4f} (F: {func_loss.item():.3f}, P: {param_loss.item():.3f}), "
                      f"Acc: {correct.item()*100:.1f}%")
        
        # Epoch summary
        avg_func_loss = epoch_func_loss / num_batches
        avg_param_loss = epoch_param_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches
        avg_acc = epoch_acc / num_batches
        elapsed = time.time() - start_time
        
        print(f"\n   ✅ Epoch {epoch+1} - Total: {avg_total_loss:.4f}, Func: {avg_func_loss:.4f}, "
              f"Param: {avg_param_loss:.4f}, Acc: {avg_acc*100:.1f}% ({elapsed/60:.1f}min)")
        
        # Save best model
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            torch.save(model.state_dict(), "/tmp/pvc_v2_param_supervised_model.pth")
            print(f"   💾 Saved best model (loss: {best_loss:.4f})")
        
        print()
        
        # Learning rate decay
        scheduler.step()
    
    total_time = time.time() - start_time
    print(f"✅ Training complete in {total_time/60:.1f} minutes")
    print()
    
    # Evaluate reconstruction quality
    print("🧪 Evaluating reconstruction quality...")
    model.eval()
    reconstructor = PVCv2Reconstructor(model, device)
    
    psnr_scores = []
    ssim_scores = []
    
    for i in range(20):
        test_frame, _ = generator.generate_anime_like_scene()
        reconstructed, funcs, size_bytes = reconstructor.reconstruct_frame(test_frame)
        metrics = reconstructor.evaluate_reconstruction(test_frame, reconstructed)
        psnr_scores.append(metrics['psnr'])
        ssim_scores.append(metrics['ssim'])
    
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    std_psnr = np.std(psnr_scores)
    std_ssim = np.std(ssim_scores)
    
    print(f"\n   📊 Quality Metrics (20 test samples):")
    print(f"      PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"      SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print()
    
    # Save comparison
    test_frame, _ = generator.generate_anime_like_scene()
    reconstructed, funcs, size_bytes = reconstructor.reconstruct_frame(test_frame)
    comparison = np.hstack([test_frame, reconstructed])
    cv2.imwrite("/tmp/pvc_v2_param_supervised_comparison.png", comparison)
    print(f"✅ Saved comparison: /tmp/pvc_v2_param_supervised_comparison.png")
    print()
    
    print("="*70)
    print("✅ Parameter-Supervised Training Complete!")
    print("="*70)
    print(f"\n📊 Final Results:")
    print(f"   Samples:       {num_samples}")
    print(f"   Epochs:        {num_epochs}")
    print(f"   Time:          {total_time/60:.1f} minutes")
    print(f"   Best Loss:     {best_loss:.4f}")
    print(f"   Final Acc:     {avg_acc*100:.1f}%")
    print(f"   Avg PSNR:      {avg_psnr:.2f} dB")
    print(f"   Avg SSIM:      {avg_ssim:.4f}")
    print()
    
    # Comparison with previous
    prev_psnr = 3.19  # From extended training without param supervision
    improvement = ((avg_psnr - prev_psnr) / prev_psnr) * 100
    
    print(f"📈 Improvement over previous:")
    print(f"   Previous PSNR: 3.19 dB")
    print(f"   New PSNR:      {avg_psnr:.2f} dB")
    print(f"   Improvement:   {improvement:.1f}%")
    print()
    
    return model


if __name__ == '__main__':
    model = train_with_parameter_supervision(num_samples=5000, num_epochs=30)

