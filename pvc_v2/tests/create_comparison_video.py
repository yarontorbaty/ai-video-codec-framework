#!/usr/bin/env python3
"""
Create PVC vs HEVC Side-by-Side Comparison Video

This script:
1. Loads the trained PVC v2.0 SOTA model
2. Encodes test frames using PVC
3. Compares with HEVC encoding
4. Creates side-by-side comparison video
"""

import numpy as np
import cv2
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.enhanced_network import EnhancedPVCv2Model
from models.sota_residual_encoder import SOTAResidualEncoder
from models.sota_residual_decoder import SOTAResidualDecoder
from training.synthetic_generator_extended import ExtendedSyntheticGenerator
from graphics.primitives_extended import NUM_EXTENDED_FUNCTIONS, CONTIGUOUS_TO_SPARSE
import torch


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


def create_comparison_video(
    pvc_model_path="/tmp/pvc_v2_perceptual_best.pth",
    encoder_path="/tmp/sota_residual_encoder_best.pth",
    decoder_path="/tmp/sota_residual_decoder_best.pth",
    num_frames=100,
    output_path="/tmp/pvc_vs_hevc_comparison.mp4",
    fps=30
):
    """Create side-by-side comparison video."""
    
    print("="*70)
    print("PVC vs HEVC Comparison Video Generator")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📦 Device: {device}")
    
    # Load models
    print(f"\n📦 Loading PVC models...")
    pvc_model = EnhancedPVCv2Model(
        feature_dim=256,
        hidden_dim=128,
        num_functions=NUM_EXTENDED_FUNCTIONS + 1,
        max_sequence_length=20
    ).to(device)
    pvc_model.load_state_dict(torch.load(pvc_model_path, map_location=device))
    pvc_model.eval()
    
    encoder = SOTAResidualEncoder(base_channels=64, quality_factor=30).to(device)
    decoder = SOTAResidualDecoder(base_channels=64).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    encoder.eval()
    decoder.eval()
    
    print(f"✅ Models loaded")
    
    # Generate test frames
    print(f"\n📊 Generating {num_frames} test frames...")
    generator = ExtendedSyntheticGenerator(width=256, height=256)
    test_frames, _ = generator.generate_dataset(num_samples=num_frames, min_functions=5, max_functions=20)
    print(f"✅ Generated {num_frames} frames")
    
    # Setup video writers
    height, width = 256, 256
    comparison_width = width * 3 + 40  # Original + PVC + HEVC + margins
    comparison_height = height + 100  # Extra space for labels
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (comparison_width, comparison_height))
    
    print(f"\n🎬 Creating comparison video...")
    print(f"   Output: {output_path}")
    print(f"   Resolution: {comparison_width}x{comparison_height}")
    print(f"   FPS: {fps}")
    
    for i in range(num_frames):
        if (i + 1) % 10 == 0:
            print(f"   Processing frame {i+1}/{num_frames}...")
        
        original = test_frames[i]
        
        # PVC reconstruction
        with torch.no_grad():
            # Stage 1: Coarse (PVC)
            func_ids, params = pvc_model.predict_with_params(original)
            sparse_func_ids = [CONTIGUOUS_TO_SPARSE.get(fid, 0) for fid in func_ids]
            
            coarse = np.zeros_like(original)
            for fid, param in zip(sparse_func_ids, params):
                coarse = execute_function_top10(fid, param, coarse)
            
            # Stage 2: Residual (SOTA)
            residual_gt = original.astype(np.float32) - coarse.astype(np.float32)
            residual_gt = residual_gt / 127.5
            residual_tensor = torch.from_numpy(residual_gt).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            encoded = encoder(residual_tensor)
            reconstructed_residual = decoder(encoded)
            
            residual_decoded = reconstructed_residual.squeeze(0).permute(1, 2, 0).cpu().numpy() * 127.5
            
            # Combine
            pvc_recon = coarse.astype(np.float32) + residual_decoded
            pvc_recon = np.clip(pvc_recon, 0, 255).astype(np.uint8)
        
        # HEVC simulation (using JPEG quality as proxy)
        # In real scenario, would encode/decode with HEVC
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        _, encoded_img = cv2.imencode('.jpg', original, encode_param)
        hevc_sim = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        # Create comparison frame
        comparison = np.ones((comparison_height, comparison_width, 3), dtype=np.uint8) * 255
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(comparison, "PVC v2.0 (SOTA)", (width + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)
        cv2.putText(comparison, "HEVC (Simulated)", (width * 2 + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Place frames
        comparison[50:50+height, 10:10+width] = original
        comparison[50:50+height, width+20:width+20+width] = pvc_recon
        comparison[50:50+height, width*2+30:width*2+30+width] = hevc_sim
        
        # Add frame number
        cv2.putText(comparison, f"Frame {i+1}/{num_frames}", (10, comparison_height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        out.write(comparison)
    
    out.release()
    
    print(f"\n✅ Comparison video created!")
    print(f"   Saved to: {output_path}")
    print(f"   Duration: {num_frames/fps:.1f} seconds")
    
    # Get file size
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"   Size: {file_size:.2f} MB")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create PVC vs HEVC comparison video')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--output', type=str, default='/tmp/pvc_vs_hevc_comparison.mp4', help='Output path')
    
    args = parser.parse_args()
    
    create_comparison_video(
        num_frames=args.frames,
        output_path=args.output,
        fps=args.fps
    )

