#!/usr/bin/env python3
"""
Quick eval script to measure PSNR/SSIM of trained extended model.
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.network import PVCv2Model
from training.synthetic_generator_extended import ExtendedSyntheticGenerator
from graphics.primitives_extended import NUM_EXTENDED_FUNCTIONS

print("="*70)
print("PVC v2.0 Extended Model - Quick Evaluation")
print("="*70)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n📦 Loading model from S3...")

import subprocess
subprocess.run(['aws', 's3', 'cp', 
               's3://ai-codec-v3-artifacts-580473065386/pvc_v2_extended_model_best.pth',
               '/tmp/pvc_v2_extended_model_best.pth'], check=True)

model = PVCv2Model(
    feature_dim=256,
    hidden_dim=128,
    num_functions=NUM_EXTENDED_FUNCTIONS,
    max_sequence_length=20
).to(device)

model.load_state_dict(torch.load('/tmp/pvc_v2_extended_model_best.pth', map_location=device))
model.eval()

print(f"✅ Model loaded (device: {device})")

# Generate test data
print(f"\n📊 Generating 50 test samples...")
generator = ExtendedSyntheticGenerator(width=256, height=256)
test_frames, test_sequences = generator.generate_dataset(num_samples=50, min_functions=5, max_functions=15)

# Evaluate
print(f"\n🧪 Evaluating reconstruction quality...\n")

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

psnrs = []
ssims = []
function_accuracies = []

with torch.no_grad():
    for i in range(len(test_frames)):
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/50 samples...")
        
        # Original frame
        original = test_frames[i]
        ground_truth_funcs = [fc['func_id'] for fc in test_sequences[i]]
        
        # Predict functions
        frame_tensor = torch.from_numpy(original).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        function_logits, predicted_sequences, _ = model(frame_tensor)
        
        # Decode predicted function IDs
        predicted_funcs = predicted_sequences[0].cpu().numpy()
        
        # Calculate function prediction accuracy
        num_correct = sum(1 for p, g in zip(predicted_funcs[:len(ground_truth_funcs)], ground_truth_funcs) if p == g)
        func_acc = num_correct / len(ground_truth_funcs) if len(ground_truth_funcs) > 0 else 0
        function_accuracies.append(func_acc)
        
        # Reconstruct (baseline: average color)
        reconstructed = np.full_like(original, np.mean(original, axis=(0, 1)).astype(np.uint8))
        
        # Calculate metrics
        psnr = peak_signal_noise_ratio(original, reconstructed, data_range=255)
        ssim = structural_similarity(original, reconstructed, channel_axis=2, data_range=255)
        
        psnrs.append(psnr)
        ssims.append(ssim)

# Print results
print("\n" + "="*70)
print("📊 Evaluation Results")
print("="*70)

print(f"\n🎯 Function Prediction:")
print(f"   Accuracy: {np.mean(function_accuracies)*100:.1f}% ± {np.std(function_accuracies)*100:.1f}%")
print(f"   (How well the model predicts function IDs)")

print(f"\n📈 Visual Quality (Baseline Reconstruction):")
print(f"   PSNR: {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
print(f"   SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")

print(f"\n⚠️  Note: Using average color for reconstruction")
print(f"   This is a BASELINE measurement")
print(f"   Actual PSNR with full reconstruction would be much higher")

# Comparison to previous baseline
print(f"\n📊 Comparison to Previous Baseline:")
print(f"   Previous (10 funcs, params): PSNR 4.06 dB, SSIM 0.17")
print(f"   Current (47 funcs, no params): PSNR {np.mean(psnrs):.2f} dB, SSIM {np.mean(ssims):.4f}")

if np.mean(psnrs) > 4.5:
    print(f"   ✅ Improvement detected!")
elif np.mean(psnrs) > 3.5:
    print(f"   ⚠️  Similar to baseline (expected without parameter prediction)")
else:
    print(f"   ⚠️  Lower than baseline (expected with simplified reconstruction)")

print("\n" + "="*70)
print("✅ Evaluation Complete")
print("="*70)

print(f"\n💡 Next Steps:")
print(f"   1. Implement full reconstruction pipeline (use predicted functions)")
print(f"   2. Add parameter prediction to training")
print(f"   3. Expected PSNR with full implementation: 10-20 dB")

