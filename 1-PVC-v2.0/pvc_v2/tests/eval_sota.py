#!/usr/bin/env python3
"""
Evaluate SOTA Hybrid Codec (32M parameters)

Compare SOTA (32M params) vs Simple (67K params)
"""

import numpy as np
import cv2
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.enhanced_network import EnhancedPVCv2Model
from models.sota_residual_encoder import SOTAResidualEncoder
from models.sota_residual_decoder import SOTAResidualDecoder
from training.synthetic_generator_extended import ExtendedSyntheticGenerator
from graphics.primitives_extended import NUM_EXTENDED_FUNCTIONS, CONTIGUOUS_TO_SPARSE
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def execute_function_top10(func_id: int, params: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """Execute graphics functions."""
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


def evaluate_sota():
    """Evaluate the SOTA trained model."""
    print("="*70)
    print("SOTA Hybrid Codec Evaluation - 32M Parameters")
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
    
    # Load SOTA residual codec
    residual_encoder = SOTAResidualEncoder(base_channels=64, quality_factor=30).to(device)
    residual_decoder = SOTAResidualDecoder(base_channels=64).to(device)
    
    residual_encoder.load_state_dict(torch.load('/tmp/sota_residual_encoder_quick.pth', map_location=device))
    residual_decoder.load_state_dict(torch.load('/tmp/sota_residual_decoder_quick.pth', map_location=device))
    
    residual_encoder.eval()
    residual_decoder.eval()
    
    encoder_params = sum(p.numel() for p in residual_encoder.parameters())
    decoder_params = sum(p.numel() for p in residual_decoder.parameters())
    
    print(f"✅ All models loaded")
    print(f"   SOTA Encoder: {encoder_params:,} params ({encoder_params*4/1024/1024:.1f} MB)")
    print(f"   SOTA Decoder: {decoder_params:,} params ({decoder_params*4/1024/1024:.1f} MB)")
    print(f"   Total: {encoder_params+decoder_params:,} params ({(encoder_params+decoder_params)*4/1024/1024:.1f} MB)")
    
    # Generate test set
    print(f"\n📊 Generating 100 test samples...")
    generator = ExtendedSyntheticGenerator(width=256, height=256)
    test_frames, _ = generator.generate_dataset(num_samples=100, min_functions=5, max_functions=20)
    
    print(f"\n🧪 Evaluating SOTA hybrid codec...\n")
    
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
        
        # Stage 2: SOTA residual encoding/decoding
        residual_gt = original.astype(np.float32) - coarse.astype(np.float32)
        residual_gt = residual_gt / 127.5
        residual_tensor = torch.from_numpy(residual_gt).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            encoded = residual_encoder(residual_tensor)
            reconstructed_residual_tensor = residual_decoder(encoded)
        
        residual_decoded = reconstructed_residual_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
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
    print("📊 FINAL RESULTS - SOTA Hybrid (32M params)")
    print("="*70)
    
    if coarse_psnrs and final_psnrs:
        avg_coarse_psnr = np.mean(coarse_psnrs)
        std_coarse_psnr = np.std(coarse_psnrs)
        
        avg_final_psnr = np.mean(final_psnrs)
        std_final_psnr = np.std(final_psnrs)
        
        improvement = avg_final_psnr - avg_coarse_psnr
        
        print(f"\n📈 Quality Progression:")
        print(f"   Coarse (PVC only): {avg_coarse_psnr:.2f} ± {std_coarse_psnr:.2f} dB")
        print(f"   Final (SOTA):      {avg_final_psnr:.2f} ± {std_final_psnr:.2f} dB")
        print(f"   Improvement:       +{improvement:.2f} dB ✨")
        
        print(f"\n   Final SSIM: {np.mean(final_ssims):.4f} ± {np.std(final_ssims):.4f}")
        
        print(f"\n📊 Comparison:")
        print(f"   Baseline (PVC only):   11.22 dB")
        print(f"   Simple Hybrid (67K):   19.91 dB")
        print(f"   SOTA Hybrid (32M):     {avg_final_psnr:.2f} dB ⭐")
        print(f"   Target:                30-40 dB")
        
        improvement_vs_simple = avg_final_psnr - 19.91
        improvement_pct = (improvement_vs_simple / 19.91) * 100
        
        print(f"\n   vs Simple: {improvement_vs_simple:+.2f} dB ({improvement_pct:+.1f}%)")
        
        if avg_final_psnr >= 25:
            print(f"\n   ✅ EXCELLENT! Significant improvement!")
            print(f"   💡 Full training (50 epochs) could reach 30-40 dB")
        elif avg_final_psnr >= 22:
            print(f"\n   ✅ GOOD! Better than simple model.")
            print(f"   💡 More training recommended for target")
        elif avg_final_psnr >= 20:
            print(f"\n   📈 MODEST IMPROVEMENT over simple.")
            print(f"   💡 Architecture validated, needs more training")
        else:
            print(f"\n   ⚠️  Similar to simple model.")
            print(f"   💡 May need architecture refinement")
        
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
            encoded = residual_encoder(residual_tensor)
            reconstructed_residual = residual_decoder(encoded)
        
        residual_decoded = reconstructed_residual.squeeze(0).permute(1, 2, 0).cpu().numpy() * 127.5
        final_sample = coarse_sample.astype(np.float32) + residual_decoded
        final_sample = np.clip(final_sample, 0, 255).astype(np.uint8)
        
        comparison = np.hstack([original_sample, coarse_sample, final_sample])
        cv2.imwrite('/tmp/sota_hybrid_comparison.png', comparison)
        print(f"   Saved: /tmp/sota_hybrid_comparison.png")
        print(f"   Layout: [Original | Coarse | SOTA Final]")
    
    print("\n" + "="*70)
    print("✅ SOTA Evaluation Complete")
    print("="*70)
    
    return np.mean(final_psnrs) if final_psnrs else None


if __name__ == "__main__":
    psnr = evaluate_sota()

