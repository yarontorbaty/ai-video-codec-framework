#!/usr/bin/env python3
"""
Quick Training Script for PVC v2.0 Proof-of-Concept

Trains the neural network on synthetic data to prove the concept works.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.network import PVCv2Model
from training.synthetic_generator import SyntheticDataGenerator
from graphics.primitives import GraphicsPrimitives, FunctionCall


class FunctionSequenceDataset(Dataset):
    """Dataset of (frame, function_sequence) pairs."""
    
    def __init__(self, frames: List[np.ndarray], function_sequences: List[List[FunctionCall]]):
        """
        Initialize dataset.
        
        Args:
            frames: List of rendered frames (uint8, H x W x 3)
            function_sequences: List of function call lists
        """
        self.frames = frames
        self.function_sequences = function_sequences
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            (frame_tensor, function_ids_tensor)
        """
        # Convert frame to tensor (3 x H x W), float32, [0, 1]
        frame = self.frames[idx]
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        # Convert function sequence to tensor
        func_ids = [f.func_id for f in self.function_sequences[idx]]
        # Pad to length 20
        while len(func_ids) < 20:
            func_ids.append(10)  # END token
        func_ids = func_ids[:20]
        func_ids_tensor = torch.tensor(func_ids, dtype=torch.long)
        
        return frame_tensor, func_ids_tensor


def train_quick_poc(num_samples: int = 500, num_epochs: int = 5):
    """
    Quick proof-of-concept training.
    
    Args:
        num_samples: Number of training samples
        num_epochs: Number of training epochs
    """
    print("="*60)
    print("PVC v2.0 - Proof of Concept Training")
    print("="*60)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    print()
    
    # Step 1: Generate synthetic data
    print("📊 Step 1: Generating synthetic training data...")
    generator = SyntheticDataGenerator(width=256, height=256)
    frames, function_sequences = generator.generate_dataset(
        num_samples=num_samples,
        anime_ratio=0.3
    )
    print()
    
    # Step 2: Create dataset and dataloader
    print("📦 Step 2: Creating dataset...")
    dataset = FunctionSequenceDataset(frames, function_sequences)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"   Dataset size: {len(dataset)} samples")
    print(f"   Batch size: 8")
    print(f"   Batches per epoch: {len(dataloader)}")
    print()
    
    # Step 3: Create model
    print("🧠 Step 3: Creating neural network...")
    model = PVCv2Model(
        feature_dim=256,
        hidden_dim=256,
        num_functions=10,
        max_sequence_length=20
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
    print()
    
    # Step 4: Setup training
    print("⚙️  Step 4: Setting up training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"   Optimizer: Adam (lr=0.001)")
    print(f"   Loss: CrossEntropyLoss")
    print()
    
    # Step 5: Train
    print(f"🚀 Step 5: Training for {num_epochs} epochs...")
    print()
    
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        for batch_idx, (frames_batch, func_ids_batch) in enumerate(dataloader):
            frames_batch = frames_batch.to(device)
            func_ids_batch = func_ids_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            function_logits, predicted_sequences, _ = model(frames_batch, func_ids_batch)
            
            # Compute loss (for each position in sequence, up to actual predicted length)
            seq_len = min(function_logits.size(1), func_ids_batch.size(1))
            loss = 0
            for t in range(seq_len):
                loss += criterion(function_logits[:, t, :], func_ids_batch[:, t])
            loss = loss / seq_len
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Compute accuracy (only for positions where predictions exist)
            predicted = torch.argmax(function_logits[:, :seq_len, :], dim=2)
            target = func_ids_batch[:, :seq_len]
            # Mask out END tokens (10) from ground truth
            mask = (target != 10).float()
            if mask.sum() > 0:
                correct = ((predicted == target).float() * mask).sum() / mask.sum()
            else:
                correct = torch.tensor(0.0)
            
            epoch_loss += loss.item()
            epoch_acc += correct.item()
            num_batches += 1
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(dataloader)} - "
                      f"Loss: {loss.item():.4f}, Acc: {correct.item()*100:.1f}%")
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        elapsed = time.time() - start_time
        print(f"\n   ✅ Epoch {epoch+1} complete - Loss: {avg_loss:.4f}, Acc: {avg_acc*100:.1f}% "
              f"(Time: {elapsed:.1f}s)\n")
    
    total_time = time.time() - start_time
    print(f"✅ Training complete in {total_time:.1f}s")
    print()
    
    # Step 6: Save model
    print("💾 Saving model...")
    model_path = "/tmp/pvc_v2_poc_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"   Saved to: {model_path}")
    print()
    
    # Step 7: Quick validation
    print("🧪 Quick validation on test sample...")
    model.eval()
    with torch.no_grad():
        # Generate a test sample
        test_frame, test_funcs = generator.generate_anime_like_scene()
        test_tensor = torch.from_numpy(test_frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        
        # Predict
        _, predicted_seq, _ = model(test_tensor)
        
        # Compare
        ground_truth = [f.func_id for f in test_funcs]
        predicted = predicted_seq[0].cpu().numpy()[:len(ground_truth)]
        
        print(f"   Ground truth: {ground_truth}")
        print(f"   Predicted:    {list(predicted)}")
        
        # Calculate accuracy
        correct = sum(1 for gt, pred in zip(ground_truth, predicted) if gt == pred)
        acc = correct / len(ground_truth) * 100
        print(f"   Sequence accuracy: {acc:.1f}%")
    
    print()
    print("="*60)
    print("✅ Proof-of-Concept Training Complete!")
    print("="*60)
    print()
    print(f"📊 Summary:")
    print(f"   Samples: {num_samples}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Time: {total_time:.1f}s")
    print(f"   Final Loss: {avg_loss:.4f}")
    print(f"   Final Accuracy: {avg_acc*100:.1f}%")
    print(f"   Model saved: {model_path}")
    print()
    
    return model


if __name__ == '__main__':
    """Run proof-of-concept training."""
    model = train_quick_poc(num_samples=500, num_epochs=5)

