#!/usr/bin/env python3
"""
Evaluate Trained Hybrid PVC v2.0 Codec

Measures final PSNR/SSIM with trained residual encoder/decoder.
Target: 30-40 dB PSNR
"""

import numpy as np
import cv2
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.enhanced_network import EnhancedPVCv2Model
from models.residual_encoder import ResidualEncoder
from models.residual_decoder import ResidualDecoder
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
        else:
            avg_color = tuple(((np.array(color1) + np.array(color2)) / 2).astype(int).tolist())
            canvas[:] = avg_color
    except:
        pass
    
    return canvas


def evaluate_hybrid():
    """Evaluate the trained hybrid codec."""
    print("="*70)
    print("Hybrid PVC v2.0 Evaluation - Trained Model")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📦 Loading models...")
    print(f"   Device: {device}")
    
    # Load PVC model
    pvc_model = EnhancedPVCv2Model(
        feature_dim=256,
        hidden_dim=128,
        num_functions=NUM_EXTENDED_FUNCTIONS + 1,
        max_sequence_length=20
    ).to(device)
    pvc_model.load_state_dict(torch.load('/tmp/pvc_v2_perceptual_best.pth', map_location=device))
    pvc_model.eval()
    
    # Load trained residual codec
    residual_encoder = ResidualEncoder(quality_factor=20).to(device)
    residual_decoder = ResidualDecoder().to(device)
    
    residual_encoder.load_state_dict(torch.load('/tmp/hybrid_residual_encoder_best.pth', map_location=device))
    residual_decoder.load_state_dict(torch.load('/tmp/hybrid_residual_decoder_best.pth', map_location=device))
    
    residual_encoder.eval()
    residual_decoder.eval()
    
    print(f"✅ All models loaded")
    
    # Generate test set
    print(f"\n📊 Generating 100 test samples...")
    generator = ExtendedSyntheticGenerator(width=256, height=256)
    test_frames, _ = generator.generate_dataset(num_samples=100, min_functions=5, max_functions=20)
    
    print(f"\n🧪 Evaluating hybrid codec...\n")
    
    coarse_psnrs = []
    final_psnrs = []
    final_ssims = []
    
    for i in range(len(test_frames)):
        if (i + 1) % 20 == 0:
            print(f"   Processed {i + 1}/100 samples...")
        
        original = test_frames[i]
        
        # Stage 1: PVC reconstruction (coarse)
        with torch.no_grad():
            func_ids, params = pvc_model.predict_with_params(original)
            sparse_func_ids = [CONTIGUOUS_TO_SPARSE.get(fid, 0) for fid in func_ids]
            
            coarse = np.zeros_like(original)
            for fid, param in zip(sparse_func_ids, params):
                coarse = execute_function_top10(fid, param, coarse)
        
        coarse_psnr = peak_signal_noise_ratio(original, coarse, data_range=255)
        if not np.isinf(coarse_psnr):
            coarse_psnrs.append(coarse_psnr)
        
        # Stage 2: Residual encoding/decoding
        residual_gt = original.astype(np.float32) - coarse.astype(np.float32)
        residual_gt = residual_gt / 127.5
        residual_tensor = torch.from_numpy(residual_gt).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            compressed = residual_encoder(residual_tensor)
            reconstructed_residual = residual_decoder(compressed)
        
        residual_decoded = reconstructed_residual.squeeze(0).permute(1, 2, 0).cpu().numpy()
        residual_decoded = residual_decoded * 127.5
        
        # Combine
        final = coarse.astype(np.float32) + residual_decoded
        final = np.clip(final, 0, 255).astype(np.uint8)
        
        final_psnr = peak_signal_noise_ratio(original, final, data_range=255)
        final_ssim = structural_similarity(original, final, channel_axis=2, data_range=255)
        
        if not np.isinf(final_psnr):
            final_psnrs.append(final_psnr)
        final_ssims.append(final_ssim)
    
    print("\n" + "="*70)
    print("📊 FINAL RESULTS - Hybrid PVC v2.0")
    print("="*70)
    
    if coarse_psnrs and final_psnrs:
        avg_coarse_psnr = np.mean(coarse_psnrs)
        std_coarse_psnr = np.std(coarse_psnrs)
        
        avg_final_psnr = np.mean(final_psnrs)
        std_final_psnr = np.std(final_psnrs)
        
        improvement = avg_final_psnr - avg_coarse_psnr
        
        print(f"\n📈 Quality Progression:")
        print(f"   Coarse (PVC only): {avg_coarse_psnr:.2f} ± {std_coarse_psnr:.2f} dB")
        print(f"   Final (Hybrid):    {avg_final_psnr:.2f} ± {std_final_psnr:.2f} dB")
        print(f"   Improvement:       +{improvement:.2f} dB ✨")
        
        print(f"\n   Final SSIM: {np.mean(final_ssims):.4f} ± {np.std(final_ssims):.4f}")
        
        print(f"\n📊 Comparison to Targets:")
        print(f"   Baseline (no hybrid): 11.22 dB")
        print(f"   Current (hybrid):     {avg_final_psnr:.2f} dB")
        print(f"   Target:               30-40 dB")
        
        if avg_final_psnr >= 30:
            print(f"\n   🎉 SUCCESS! Achieved 30-40 dB target!")
            print(f"   🏆 Production-quality codec!")
        elif avg_final_psnr >= 20:
            print(f"\n   ✅ GOOD PROGRESS! Approaching target.")
            print(f"   💡 Further training could reach 30 dB")
        elif avg_final_psnr >= 15:
            print(f"\n   📈 IMPROVEMENT! Better than baseline.")
            print(f"   💡 Need more training or higher quality factor")
        else:
            print(f"\n   ⚠️  Modest improvement from baseline.")
            print(f"   💡 Consider: more epochs, higher quality, more data")
        
        # Save comparison
        print(f"\n💾 Saving sample comparison...")
        original_sample = test_frames[0]
        
        func_ids, params = pvc_model.predict_with_params(original_sample)
        sparse_func_ids = [CONTIGUOUS_TO_SPARSE.get(fid, 0) for fid in func_ids]
        coarse_sample = np.zeros_like(original_sample)
        for fid, param in zip(sparse_func_ids, params):
            coarse_sample = execute_function_top10(fid, param, coarse_sample)
        
        residual_gt = original_sample.astype(np.float32) - coarse_sample.astype(np.float32)
        residual_gt = residual_gt / 127.5
        residual_tensor = torch.from_numpy(residual_gt).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            compressed = residual_encoder(residual_tensor)
            reconstructed_residual = residual_decoder(compressed)
        
        residual_decoded = reconstructed_residual.squeeze(0).permute(1, 2, 0).cpu().numpy() * 127.5
        final_sample = coarse_sample.astype(np.float32) + residual_decoded
        final_sample = np.clip(final_sample, 0, 255).astype(np.uint8)
        
        comparison = np.hstack([original_sample, coarse_sample, final_sample])
        cv2.imwrite('/tmp/hybrid_pvc_trained_comparison.png', comparison)
        print(f"   Saved: /tmp/hybrid_pvc_trained_comparison.png")
        print(f"   Layout: [Original | Coarse | Final]")
    
    print("\n" + "="*70)
    print("✅ Evaluation Complete")
    print("="*70)
    
    return np.mean(final_psnrs) if final_psnrs else None


if __name__ == "__main__":
    psnr = evaluate_hybrid()

