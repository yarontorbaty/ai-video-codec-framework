#!/usr/bin/env python3
"""
Extended Training for PVC v2.0

Train with more data and epochs to improve quality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
from pathlib import Path
import time
import cv2

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.network import PVCv2Model
from models.reconstructor import PVCv2Reconstructor
from training.synthetic_generator import SyntheticDataGenerator
from graphics.primitives import FunctionCall
from typing import List, Tuple


class FunctionSequenceDataset(Dataset):
    """Dataset of (frame, function_sequence) pairs."""
    
    def __init__(self, frames: List[np.ndarray], function_sequences: List[List[FunctionCall]]):
        self.frames = frames
        self.function_sequences = function_sequences
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert frame to tensor
        frame = self.frames[idx]
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        # Convert function sequence to tensor
        func_ids = [f.func_id for f in self.function_sequences[idx]]
        while len(func_ids) < 20:
            func_ids.append(10)  # END token
        func_ids = func_ids[:20]
        func_ids_tensor = torch.tensor(func_ids, dtype=torch.long)
        
        return frame_tensor, func_ids_tensor


def train_extended(num_samples: int = 2000, num_epochs: int = 20):
    """
    Extended training with more data and epochs.
    
    Args:
        num_samples: Number of training samples
        num_epochs: Number of training epochs
    """
    print("="*60)
    print("PVC v2.0 - Extended Training")
    print("="*60)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    print()
    
    # Generate training data
    print(f"📊 Generating {num_samples} training samples...")
    generator = SyntheticDataGenerator(width=256, height=256)
    frames, function_sequences = generator.generate_dataset(
        num_samples=num_samples,
        anime_ratio=0.5  # More anime-like samples
    )
    print()
    
    # Create dataset
    print("📦 Creating dataset...")
    dataset = FunctionSequenceDataset(frames, function_sequences)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # Larger batch
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
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    print(f"   Optimizer: Adam (lr=0.001, decay every 5 epochs)")
    print()
    
    # Train
    print(f"🚀 Training for {num_epochs} epochs...")
    print()
    
    model.train()
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        for batch_idx, (frames_batch, func_ids_batch) in enumerate(dataloader):
            frames_batch = frames_batch.to(device)
            func_ids_batch = func_ids_batch.to(device)
            
            # Forward
            optimizer.zero_grad()
            function_logits, predicted_sequences, _ = model(frames_batch, func_ids_batch)
            
            # Loss
            seq_len = min(function_logits.size(1), func_ids_batch.size(1))
            loss = 0
            for t in range(seq_len):
                loss += criterion(function_logits[:, t, :], func_ids_batch[:, t])
            loss = loss / seq_len
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Accuracy
            predicted = torch.argmax(function_logits[:, :seq_len, :], dim=2)
            target = func_ids_batch[:, :seq_len]
            mask = (target != 10).float()
            if mask.sum() > 0:
                correct = ((predicted == target).float() * mask).sum() / mask.sum()
            else:
                correct = torch.tensor(0.0)
            
            epoch_loss += loss.item()
            epoch_acc += correct.item()
            num_batches += 1
            
            # Print progress
            if (batch_idx + 1) % 20 == 0:
                print(f"   Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(dataloader)} - "
                      f"Loss: {loss.item():.4f}, Acc: {correct.item()*100:.1f}%")
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        elapsed = time.time() - start_time
        
        print(f"\n   ✅ Epoch {epoch+1} complete - Loss: {avg_loss:.4f}, Acc: {avg_acc*100:.1f}% "
              f"(Time: {elapsed:.1f}s)")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "/tmp/pvc_v2_extended_model.pth")
            print(f"   💾 Saved best model (loss: {best_loss:.4f})")
        
        print()
        
        # Learning rate decay
        scheduler.step()
    
    total_time = time.time() - start_time
    print(f"✅ Training complete in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print()
    
    # Test reconstruction quality
    print("🧪 Testing reconstruction quality...")
    model.eval()
    reconstructor = PVCv2Reconstructor(model, device)
    
    # Test on multiple samples
    psnr_scores = []
    ssim_scores = []
    
    for i in range(10):
        test_frame, _ = generator.generate_anime_like_scene()
        reconstructed, funcs, size_bytes = reconstructor.reconstruct_frame(test_frame)
        metrics = reconstructor.evaluate_reconstruction(test_frame, reconstructed)
        psnr_scores.append(metrics['psnr'])
        ssim_scores.append(metrics['ssim'])
    
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    
    print(f"\n   Average PSNR: {avg_psnr:.2f} dB")
    print(f"   Average SSIM: {avg_ssim:.4f}")
    print()
    
    # Save final comparison
    test_frame, _ = generator.generate_anime_like_scene()
    reconstructed, funcs, size_bytes = reconstructor.reconstruct_frame(test_frame)
    comparison = np.hstack([test_frame, reconstructed])
    cv2.imwrite("/tmp/pvc_v2_extended_comparison.png", comparison)
    print(f"✅ Saved comparison: /tmp/pvc_v2_extended_comparison.png")
    print()
    
    print("="*60)
    print("✅ Extended Training Complete!")
    print("="*60)
    print(f"\n📊 Final Results:")
    print(f"   Samples:  {num_samples}")
    print(f"   Epochs:   {num_epochs}")
    print(f"   Time:     {total_time/60:.1f} minutes")
    print(f"   Loss:     {best_loss:.4f}")
    print(f"   Accuracy: {avg_acc*100:.1f}%")
    print(f"   Avg PSNR: {avg_psnr:.2f} dB")
    print(f"   Avg SSIM: {avg_ssim:.4f}")
    print()


if __name__ == '__main__':
    train_extended(num_samples=2000, num_epochs=20)

