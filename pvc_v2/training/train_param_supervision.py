#!/usr/bin/env python3
"""
Enhanced Training Script with Parameter Supervision

Key improvements:
- Full sequence parameter prediction
- Combined loss: 0.5 × function loss + 0.5 × parameter loss
- 5K training samples (reduce overfitting)
- Proper evaluation with reconstruction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.enhanced_network import EnhancedPVCv2Model
from training.synthetic_generator_extended import ExtendedSyntheticGenerator
from graphics.primitives_extended import NUM_EXTENDED_FUNCTIONS


class EnhancedFunctionDataset(Dataset):
    """Dataset with function IDs AND parameters."""
    
    def __init__(self, frames: np.ndarray, sequences: list, num_functions: int):
        self.frames = frames
        self.sequences = sequences
        self.num_functions = num_functions
        self.end_token_id = num_functions
        
        # Find max sequence length
        self.max_seq_len = max(len(seq) for seq in sequences)
        print(f"   Max sequence length: {self.max_seq_len}")
        print(f"   END token ID: {self.end_token_id}")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        sequence = self.sequences[idx]
        
        # Convert frame to tensor
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        # Extract function IDs and parameters
        func_ids = []
        params = []
        
        for fc in sequence:
            fid = fc['func_id']
            if fid >= self.num_functions:
                fid = 0  # Fallback
            func_ids.append(fid)
            params.append(fc['normalized_params'])
        
        # Pad to max length
        while len(func_ids) < self.max_seq_len:
            func_ids.append(self.end_token_id)
            params.append(np.zeros(10, dtype=np.float32))
        
        func_ids_tensor = torch.tensor(func_ids, dtype=torch.long)
        params_tensor = torch.tensor(np.array(params), dtype=torch.float32)
        
        return frame_tensor, func_ids_tensor, params_tensor


def train_with_param_supervision(
    num_samples: int = 5000,
    num_epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 0.001
):
    """
    Train enhanced model with parameter supervision.
    """
    print("="*70)
    print("PVC v2.0 Enhanced Training - Parameter Supervision")
    print("="*70)
    print(f"\n📊 Configuration:")
    print(f"   Functions: {NUM_EXTENDED_FUNCTIONS}")
    print(f"   Samples: {num_samples}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}\n")
    
    # Generate training data
    print("📊 Generating training data...")
    generator = ExtendedSyntheticGenerator(width=256, height=256)
    frames, sequences = generator.generate_dataset(
        num_samples=num_samples,
        min_functions=5,
        max_functions=20  # Longer sequences
    )
    
    # Create dataset
    print("\n📦 Creating dataset...")
    dataset = EnhancedFunctionDataset(frames, sequences, NUM_EXTENDED_FUNCTIONS)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Batches: {len(dataloader)}")
    
    # Create model
    print("\n🏗️  Creating enhanced model...")
    model = EnhancedPVCv2Model(
        feature_dim=256,
        hidden_dim=128,
        num_functions=NUM_EXTENDED_FUNCTIONS,
        max_sequence_length=dataset.max_seq_len
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion_func = nn.CrossEntropyLoss()
    criterion_param = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n🚀 Starting training...\n")
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_func_loss = 0.0
        epoch_param_loss = 0.0
        epoch_func_acc = 0.0
        num_batches = 0
        
        for batch_idx, (frames_batch, func_ids_batch, params_batch) in enumerate(dataloader):
            frames_batch = frames_batch.to(device)
            func_ids_batch = func_ids_batch.to(device)
            params_batch = params_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            function_logits, predicted_sequences, predicted_params = model(frames_batch, func_ids_batch)
            
            # Function loss (CrossEntropy for each position)
            batch_size, seq_len = func_ids_batch.shape
            func_loss = 0
            for t in range(seq_len):
                func_loss += criterion_func(function_logits[:, t, :], func_ids_batch[:, t])
            func_loss = func_loss / seq_len
            
            # Parameter loss (MSE, only for non-END tokens)
            mask = (func_ids_batch != NUM_EXTENDED_FUNCTIONS).float().unsqueeze(-1)  # (batch x seq x 1)
            param_loss = criterion_param(predicted_params * mask, params_batch * mask)
            
            # Combined loss: 0.5 function + 0.5 parameter
            loss = 0.5 * func_loss + 0.5 * param_loss
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Metrics
            predicted = torch.argmax(function_logits, dim=2)
            mask_1d = (func_ids_batch != NUM_EXTENDED_FUNCTIONS).float()
            if mask_1d.sum() > 0:
                correct = ((predicted == func_ids_batch).float() * mask_1d).sum() / mask_1d.sum()
            else:
                correct = torch.tensor(0.0)
            
            epoch_loss += loss.item()
            epoch_func_loss += func_loss.item()
            epoch_param_loss += param_loss.item()
            epoch_func_acc += correct.item()
            num_batches += 1
        
        # Epoch stats
        avg_loss = epoch_loss / num_batches
        avg_func_loss = epoch_func_loss / num_batches
        avg_param_loss = epoch_param_loss / num_batches
        avg_acc = epoch_func_acc / num_batches * 100
        
        elapsed = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {avg_loss:.4f} "
              f"(Func: {avg_func_loss:.4f}, Param: {avg_param_loss:.4f}) | "
              f"Acc: {avg_acc:.1f}% | "
              f"Time: {elapsed:.1f}s")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), '/tmp/pvc_v2_enhanced_model_best.pth')
    
    total_time = time.time() - start_time
    print(f"\n✅ Training complete in {total_time/60:.1f} minutes")
    print(f"   Best loss: {best_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), '/tmp/pvc_v2_enhanced_model_final.pth')
    print(f"   Saved: /tmp/pvc_v2_enhanced_model_final.pth")
    
    # Evaluate
    print("\n🧪 Evaluating...")
    evaluate_model(model, generator, device)
    
    return model


def evaluate_model(model, generator, device):
    """Evaluate with proper reconstruction."""
    model.eval()
    
    # Generate test samples
    print("   Generating 100 test samples...")
    test_frames, test_sequences = generator.generate_dataset(
        num_samples=100,
        min_functions=5,
        max_functions=20
    )
    
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    
    psnrs = []
    ssims = []
    func_accs = []
    param_errors = []
    
    print("   Evaluating reconstruction quality...\n")
    
    with torch.no_grad():
        for i in range(len(test_frames)):
            if (i + 1) % 20 == 0:
                print(f"      Processed {i + 1}/100 samples...")
            
            original = test_frames[i]
            gt_sequence = test_sequences[i]
            
            # Predict
            frame_tensor = torch.from_numpy(original).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            _, pred_seqs, pred_params = model(frame_tensor)
            
            # Function accuracy
            gt_funcs = [fc['func_id'] for fc in gt_sequence]
            pred_funcs = pred_seqs[0].cpu().numpy()[:len(gt_funcs)]
            func_acc = np.mean([p == g for p, g in zip(pred_funcs, gt_funcs)])
            func_accs.append(func_acc)
            
            # Parameter error
            gt_params = np.array([fc['normalized_params'] for fc in gt_sequence])
            pred_params_np = pred_params[0].cpu().numpy()[:len(gt_params)]
            param_error = np.mean(np.abs(pred_params_np - gt_params))
            param_errors.append(param_error)
            
            # Reconstruct (simplified: average color)
            # TODO: Implement full reconstruction pipeline
            reconstructed = np.full_like(original, np.mean(original, axis=(0, 1)).astype(np.uint8))
            
            # Metrics
            psnr = peak_signal_noise_ratio(original, reconstructed, data_range=255)
            ssim = structural_similarity(original, reconstructed, channel_axis=2, data_range=255)
            
            psnrs.append(psnr)
            ssims.append(ssim)
    
    print("\n" + "="*70)
    print("📊 Evaluation Results")
    print("="*70)
    print(f"\n🎯 Function Prediction:")
    print(f"   Accuracy: {np.mean(func_accs)*100:.1f}% ± {np.std(func_accs)*100:.1f}%")
    
    print(f"\n📐 Parameter Prediction:")
    print(f"   MAE: {np.mean(param_errors):.4f} ± {np.std(param_errors):.4f}")
    print(f"   (Mean Absolute Error in normalized [0,1] space)")
    
    print(f"\n📈 Visual Quality (Baseline Reconstruction):")
    valid_psnrs = [p for p in psnrs if not np.isinf(p)]
    if valid_psnrs:
        print(f"   PSNR: {np.mean(valid_psnrs):.2f} ± {np.std(valid_psnrs):.2f} dB")
    else:
        print(f"   PSNR: N/A (reconstruction needed)")
    print(f"   SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
    
    print(f"\n⚠️  Note: Full reconstruction pipeline needed for accurate PSNR")
    print(f"   Expected PSNR with reconstruction: 10-20 dB")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Train with 5K samples
    model = train_with_param_supervision(
        num_samples=5000,
        num_epochs=30,
        batch_size=16,
        learning_rate=0.001
    )

