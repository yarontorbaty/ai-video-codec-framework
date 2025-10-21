#!/usr/bin/env python3
"""
Evaluate PVC v2.0 Model Trained with Perceptual Loss

Measures final PSNR/SSIM with the perceptually-trained model.
"""

import numpy as np
import cv2
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.enhanced_network import EnhancedPVCv2Model
from training.synthetic_generator_extended import ExtendedSyntheticGenerator
from graphics.primitives_extended import NUM_EXTENDED_FUNCTIONS, CONTIGUOUS_TO_SPARSE
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def execute_function_top10(func_id: int, params: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """Execute top 10 graphics functions."""
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
        elif func_id == 6:
            for c in range(3):
                gradient = np.linspace(color1[c], color2[c], height)
                canvas[:, :, c] = np.tile(gradient.reshape(-1, 1), (1, width)).astype(np.uint8)
        elif func_id == 2:
            rx = max(5, w // 2)
            ry = max(5, h // 2)
            cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, color1, -1)
        elif func_id == 13:
            square_size = max(8, min(32, w // 4))
            y_grid, x_grid = np.ogrid[:height, :width]
            pattern = ((x_grid // square_size) + (y_grid // square_size)) % 2
            for c in range(3):
                canvas[:, :, c] = np.where(pattern == 0, color1[c], color2[c]).astype(np.uint8)
        elif func_id == 23:
            if x2 > x1 and y2 > y1:
                radius = min(10, w // 4, h // 4)
                cv2.rectangle(canvas, (x1 + radius, y1), (x2 - radius, y2), color1, -1)
                cv2.rectangle(canvas, (x1, y1 + radius), (x2, y2 - radius), color1, -1)
                if radius > 0:
                    cv2.circle(canvas, (x1 + radius, y1 + radius), radius, color1, -1)
                    cv2.circle(canvas, (x2 - radius, y1 + radius), radius, color1, -1)
                    cv2.circle(canvas, (x1 + radius, y2 - radius), radius, color1, -1)
                    cv2.circle(canvas, (x2 - radius, y2 - radius), radius, color1, -1)
        elif func_id == 24:
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
            avg_color = tuple(((np.array(color1) + np.array(color2)) / 2).astype(int).tolist())
            canvas[:] = avg_color
    except:
        pass
    
    return canvas


def reconstruct(model, frame, device):
    """Reconstruct frame using model predictions."""
    model.eval()
    with torch.no_grad():
        func_ids, params = model.predict_with_params(frame)
        
        # Convert contiguous IDs to sparse IDs
        sparse_func_ids = [CONTIGUOUS_TO_SPARSE.get(fid, 0) for fid in func_ids]
        
        canvas = np.zeros_like(frame)
        for fid, param in zip(sparse_func_ids, params):
            canvas = execute_function_top10(fid, param, canvas)
        
        return canvas


def evaluate():
    """Evaluate the perceptually-trained model."""
    print("="*70)
    print("PVC v2.0 Evaluation - Model Trained with Perceptual Loss")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📦 Loading model...")
    print(f"   Device: {device}")
    
    model = EnhancedPVCv2Model(
        feature_dim=256,
        hidden_dim=128,
        num_functions=NUM_EXTENDED_FUNCTIONS + 1,
        max_sequence_length=20
    ).to(device)
    
    model.load_state_dict(torch.load('/tmp/pvc_v2_perceptual_best.pth', map_location=device))
    print(f"✅ Model loaded")
    
    print(f"\n📊 Generating 100 test samples...")
    generator = ExtendedSyntheticGenerator(width=256, height=256)
    test_frames, test_sequences = generator.generate_dataset(
        num_samples=100,
        min_functions=5,
        max_functions=20
    )
    
    print(f"\n🧪 Evaluating...\n")
    
    psnrs = []
    ssims = []
    
    for i in range(len(test_frames)):
        if (i + 1) % 20 == 0:
            print(f"   Processed {i + 1}/100 samples...")
        
        original = test_frames[i]
        reconstructed = reconstruct(model, original, device)
        
        psnr = peak_signal_noise_ratio(original, reconstructed, data_range=255)
        ssim = structural_similarity(original, reconstructed, channel_axis=2, data_range=255)
        
        if not np.isinf(psnr):
            psnrs.append(psnr)
        ssims.append(ssim)
    
    print("\n" + "="*70)
    print("📊 FINAL RESULTS - Perceptual Loss Training")
    print("="*70)
    
    if psnrs:
        avg_psnr = np.mean(psnrs)
        std_psnr = np.std(psnrs)
        
        print(f"\n📈 Visual Quality:")
        print(f"   PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
        print(f"   SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
        
        print(f"\n📊 Comparison:")
        print(f"   Baseline (no params): 4.06 dB")
        print(f"   Smoke test (10 funcs): 11.22 dB")
        print(f"   Complete (47 funcs): 10.71 dB")
        print(f"   With Perceptual Loss: {avg_psnr:.2f} dB ← NEW!")
        
        improvement = ((avg_psnr - 11.22) / 11.22) * 100
        
        print(f"\n   Improvement from smoke test: {improvement:+.1f}%")
        print(f"   Improvement from baseline: {((avg_psnr - 4.06) / 4.06) * 100:+.1f}%")
        
        if avg_psnr >= 15:
            print(f"\n   🎉 EXCELLENT! Achieved 15-18 dB target!")
        elif avg_psnr >= 12:
            print(f"\n   ✅ GOOD! Above smoke test baseline.")
        else:
            print(f"\n   ⚠️  Similar to baseline.")
        
        # Save comparison
        print(f"\n💾 Saving sample reconstruction...")
        reconstructed_sample = reconstruct(model, test_frames[0], device)
        comparison = np.hstack([test_frames[0], reconstructed_sample])
        cv2.imwrite('/tmp/pvc_v2_perceptual_result.png', comparison)
        print(f"   Saved: /tmp/pvc_v2_perceptual_result.png")
    
    print("\n" + "="*70)
    print("✅ Evaluation Complete")
    print("="*70)
    
    return np.mean(psnrs) if psnrs else None


if __name__ == "__main__":
    psnr = evaluate()

