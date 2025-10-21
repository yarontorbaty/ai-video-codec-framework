#!/usr/bin/env python3
"""
Quick Smoke Test: Reconstruction Pipeline for Top 10 Functions

Implements executors for the most common functions to get a rough PSNR estimate.
"""

import numpy as np
import cv2
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.enhanced_network import EnhancedPVCv2Model
from training.synthetic_generator_extended import ExtendedSyntheticGenerator
from graphics.primitives_extended import NUM_EXTENDED_FUNCTIONS


def execute_function(func_id: int, params: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """
    Execute a single graphics function with predicted parameters.
    
    Args:
        func_id: Function ID (0-41)
        params: Normalized parameters [10] in [0, 1]
        canvas: Current canvas (H x W x 3), uint8
        
    Returns:
        Updated canvas
    """
    height, width = canvas.shape[:2]
    
    # Denormalize parameters
    x1 = int(np.clip(params[0] * width, 0, width - 1))
    y1 = int(np.clip(params[1] * height, 0, height - 1))
    x2 = int(np.clip(params[2] * width, 0, width - 1))
    y2 = int(np.clip(params[3] * height, 0, height - 1))
    
    color1 = tuple((params[4:7] * 255).astype(int).tolist())
    color2 = tuple((params[7:10] * 255).astype(int).tolist())
    
    # Ensure valid coordinates
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Top 10 most common functions (covering ~80% of use cases)
    try:
        if func_id == 0:  # fill_solid
            canvas[:] = color1
            
        elif func_id == 3:  # draw_rect
            if x2 > x1 and y2 > y1:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color1, -1)
        
        elif func_id == 5:  # draw_circle
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            radius = max(5, min(abs(x2 - x1), abs(y2 - y1)) // 2)
            cv2.circle(canvas, (cx, cy), radius, color1, -1)
        
        elif func_id == 10:  # fill_radial_gradient
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            radius = max(10, (abs(x2 - x1) + abs(y2 - y1)) // 2)
            
            # Create radial gradient manually (simplified)
            y_coords, x_coords = np.ogrid[:height, :width]
            dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            dist_norm = np.clip(dist / max(radius, 1), 0, 1)
            
            for c in range(3):
                canvas[:, :, c] = (
                    color1[c] * (1 - dist_norm) + 
                    color2[c] * dist_norm
                ).astype(np.uint8)
        
        elif func_id == 1 or func_id == 7:  # draw_gradient_linear / horizontal
            for c in range(3):
                gradient = np.linspace(color1[c], color2[c], width)
                canvas[:, :, c] = np.tile(gradient, (height, 1)).astype(np.uint8)
        
        elif func_id == 6:  # draw_gradient_vertical
            for c in range(3):
                gradient = np.linspace(color1[c], color2[c], height)
                canvas[:, :, c] = np.tile(gradient.reshape(-1, 1), (1, width)).astype(np.uint8)
        
        elif func_id == 2:  # draw_ellipse
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            rx = max(5, abs(x2 - x1) // 2)
            ry = max(5, abs(y2 - y1) // 2)
            cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, color1, -1)
        
        elif func_id == 13:  # fill_checkerboard
            square_size = max(8, min(32, (x2 - x1) // 4))
            y_grid, x_grid = np.ogrid[:height, :width]
            pattern = ((x_grid // square_size) + (y_grid // square_size)) % 2
            
            for c in range(3):
                canvas[:, :, c] = np.where(
                    pattern == 0,
                    color1[c],
                    color2[c]
                ).astype(np.uint8)
        
        elif func_id == 23:  # draw_rounded_rect
            if x2 > x1 and y2 > y1:
                radius = min(10, (x2 - x1) // 4, (y2 - y1) // 4)
                
                # Main rectangles
                cv2.rectangle(canvas, (x1 + radius, y1), (x2 - radius, y2), color1, -1)
                cv2.rectangle(canvas, (x1, y1 + radius), (x2, y2 - radius), color1, -1)
                
                # Corners
                if radius > 0:
                    cv2.circle(canvas, (x1 + radius, y1 + radius), radius, color1, -1)
                    cv2.circle(canvas, (x2 - radius, y1 + radius), radius, color1, -1)
                    cv2.circle(canvas, (x1 + radius, y2 - radius), radius, color1, -1)
                    cv2.circle(canvas, (x2 - radius, y2 - radius), radius, color1, -1)
        
        elif func_id == 24:  # draw_star
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            outer_r = max(10, min(abs(x2 - x1), abs(y2 - y1)) // 2)
            inner_r = outer_r // 2
            
            # 5-pointed star
            pts = []
            for i in range(10):
                angle = i * np.pi / 5 - np.pi / 2
                radius = outer_r if i % 2 == 0 else inner_r
                x = int(cx + radius * np.cos(angle))
                y = int(cy + radius * np.sin(angle))
                pts.append((x, y))
            
            pts_array = np.array(pts, dtype=np.int32)
            cv2.fillPoly(canvas, [pts_array], color1)
        
        # For any other function, just fill with average color (fallback)
        else:
            avg_color = tuple(((np.array(color1) + np.array(color2)) / 2).astype(int).tolist())
            canvas[:] = avg_color
            
    except Exception as e:
        # If anything fails, just skip this function
        pass
    
    return canvas


def reconstruct_from_predictions(model, frame, device):
    """
    Reconstruct frame using predicted functions and parameters.
    
    Args:
        model: Trained EnhancedPVCv2Model
        frame: Original frame (H x W x 3), uint8
        device: torch device
        
    Returns:
        reconstructed: Reconstructed frame
        func_ids: List of function IDs
        params: List of parameter arrays
    """
    model.eval()
    
    with torch.no_grad():
        # Get predictions
        func_ids, params = model.predict_with_params(frame)
        
        # Initialize canvas (black)
        canvas = np.zeros_like(frame)
        
        # Execute each function
        for fid, param in zip(func_ids, params):
            canvas = execute_function(fid, param, canvas)
        
        return canvas, func_ids, params


def smoke_test():
    """Run quick smoke test on enhanced model."""
    print("="*70)
    print("PVC v2.0 Enhanced Model - Smoke Test")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📦 Loading model...")
    
    # Load model
    model = EnhancedPVCv2Model(
        feature_dim=256,
        hidden_dim=128,
        num_functions=NUM_EXTENDED_FUNCTIONS,
        max_sequence_length=20
    ).to(device)
    
    model.load_state_dict(torch.load('/tmp/pvc_v2_enhanced_model_best.pth', map_location=device))
    print(f"✅ Model loaded (device: {device})")
    
    # Generate test data
    print(f"\n📊 Generating 50 test samples...")
    generator = ExtendedSyntheticGenerator(width=256, height=256)
    test_frames, test_sequences = generator.generate_dataset(
        num_samples=50,
        min_functions=5,
        max_functions=20
    )
    
    # Evaluate
    print(f"\n🧪 Evaluating with reconstruction...\n")
    
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    
    psnrs = []
    ssims = []
    func_accs = []
    param_maes = []
    
    for i in range(len(test_frames)):
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/50 samples...")
        
        original = test_frames[i]
        gt_sequence = test_sequences[i]
        
        # Reconstruct
        reconstructed, pred_funcs, pred_params = reconstruct_from_predictions(model, original, device)
        
        # Function accuracy
        gt_funcs = [fc['func_id'] for fc in gt_sequence]
        min_len = min(len(pred_funcs), len(gt_funcs))
        func_acc = np.mean([p == g for p, g in zip(pred_funcs[:min_len], gt_funcs[:min_len])]) if min_len > 0 else 0
        func_accs.append(func_acc)
        
        # Parameter MAE
        gt_params = np.array([fc['normalized_params'] for fc in gt_sequence])
        pred_params_arr = np.array(pred_params)
        min_len = min(len(pred_params_arr), len(gt_params))
        if min_len > 0:
            param_mae = np.mean(np.abs(pred_params_arr[:min_len] - gt_params[:min_len]))
            param_maes.append(param_mae)
        
        # Visual metrics
        psnr = peak_signal_noise_ratio(original, reconstructed, data_range=255)
        ssim = structural_similarity(original, reconstructed, channel_axis=2, data_range=255)
        
        if not np.isinf(psnr):
            psnrs.append(psnr)
        ssims.append(ssim)
    
    # Results
    print("\n" + "="*70)
    print("📊 Smoke Test Results")
    print("="*70)
    
    print(f"\n🎯 Function Prediction:")
    print(f"   Accuracy: {np.mean(func_accs)*100:.1f}% ± {np.std(func_accs)*100:.1f}%")
    
    print(f"\n📐 Parameter Prediction:")
    if param_maes:
        print(f"   MAE: {np.mean(param_maes):.4f} ± {np.std(param_maes):.4f}")
    
    print(f"\n📈 Visual Quality (with Reconstruction):")
    if psnrs:
        avg_psnr = np.mean(psnrs)
        std_psnr = np.std(psnrs)
        print(f"   PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
        print(f"   SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
        
        # Comparison
        baseline_psnr = 4.06
        improvement = ((avg_psnr - baseline_psnr) / baseline_psnr) * 100
        
        print(f"\n📊 Comparison to Baseline:")
        print(f"   Baseline (10 funcs, no params): {baseline_psnr:.2f} dB")
        print(f"   Current (47 funcs, with params): {avg_psnr:.2f} dB")
        print(f"   Improvement: {improvement:+.1f}%")
        
        if avg_psnr >= 10:
            print(f"\n   ✅ SUCCESS! Achieved target PSNR (>10 dB)")
        elif avg_psnr >= 6:
            print(f"\n   ⚠️  Moderate improvement. Consider full implementation.")
        else:
            print(f"\n   ❌ Limited improvement. May need hybrid approach.")
    else:
        print(f"   PSNR: N/A (all infinite)")
    
    # Save sample reconstruction
    print(f"\n💾 Saving sample reconstruction...")
    original_sample = test_frames[0]
    reconstructed_sample, _, _ = reconstruct_from_predictions(model, original_sample, device)
    
    comparison = np.hstack([original_sample, reconstructed_sample])
    cv2.imwrite('/tmp/pvc_v2_reconstruction_sample.png', comparison)
    print(f"   Saved: /tmp/pvc_v2_reconstruction_sample.png")
    
    print("\n" + "="*70)
    print("✅ Smoke Test Complete")
    print("="*70)
    
    return np.mean(psnrs) if psnrs else None


if __name__ == "__main__":
    psnr = smoke_test()

