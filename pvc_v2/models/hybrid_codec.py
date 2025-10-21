#!/usr/bin/env python3
"""
Hybrid PVC v2.0 Codec

Combines procedural encoding (structure) with residual encoding (details):
- Stage 1: PVC v2.0 for coarse reconstruction (11 dB, 95% compression)
- Stage 2: Residual codec for fine details (+ 20-30 dB, 5-10% additional)

Target: 30-40 dB PSNR with 90-95% total compression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.enhanced_network import EnhancedPVCv2Model
from models.residual_encoder import ResidualEncoder
from models.residual_decoder import ResidualDecoder
from graphics.primitives_extended import NUM_EXTENDED_FUNCTIONS, CONTIGUOUS_TO_SPARSE
import cv2


class HybridPVCCodec(nn.Module):
    """
    Hybrid codec combining procedural and residual encoding.
    
    Encoding:
    1. Use PVC v2.0 to get coarse reconstruction
    2. Compute residual = original - coarse
    3. Compress residual using residual encoder
    
    Decoding:
    1. Decode procedural functions/params to coarse frame
    2. Decode compressed residuals
    3. Combine: final = coarse + residuals
    """
    
    def __init__(self, pvc_model_path=None, residual_quality=20):
        """
        Initialize hybrid codec.
        
        Args:
            pvc_model_path: Path to trained PVC v2.0 model
            residual_quality: Quality factor for residual compression (10-50)
        """
        super(HybridPVCCodec, self).__init__()
        
        # Stage 1: Procedural codec (PVC v2.0)
        self.pvc_model = EnhancedPVCv2Model(
            feature_dim=256,
            hidden_dim=128,
            num_functions=NUM_EXTENDED_FUNCTIONS + 1,
            max_sequence_length=20
        )
        
        if pvc_model_path:
            self.pvc_model.load_state_dict(torch.load(pvc_model_path, map_location='cpu'))
        
        self.pvc_model.eval()  # Frozen during residual training
        
        # Stage 2: Residual codec
        self.residual_encoder = ResidualEncoder(quality_factor=residual_quality)
        self.residual_decoder = ResidualDecoder()
        
        self.residual_quality = residual_quality
    
    def execute_function_top10(self, func_id: int, params: np.ndarray, canvas: np.ndarray) -> np.ndarray:
        """Execute top 10 graphics functions (same as training)."""
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
    
    def pvc_reconstruct(self, frame):
        """Reconstruct frame using PVC v2.0 model."""
        with torch.no_grad():
            func_ids, params = self.pvc_model.predict_with_params(frame)
            
            # Convert contiguous IDs to sparse IDs
            sparse_func_ids = [CONTIGUOUS_TO_SPARSE.get(fid, 0) for fid in func_ids]
            
            # Reconstruct
            canvas = np.zeros_like(frame)
            for fid, param in zip(sparse_func_ids, params):
                canvas = self.execute_function_top10(fid, param, canvas)
            
            return canvas
    
    def encode(self, frame):
        """
        Encode frame using hybrid approach.
        
        Args:
            frame: Input frame (H, W, 3) uint8 [0, 255]
        
        Returns:
            Dict with:
            - pvc_data: Function IDs and parameters
            - residual_data: Compressed residuals
            - coarse_psnr: PSNR of coarse reconstruction
        """
        # Stage 1: Procedural encoding
        func_ids, params = self.pvc_model.predict_with_params(frame)
        coarse = self.pvc_reconstruct(frame)
        
        # Calculate coarse PSNR
        from skimage.metrics import peak_signal_noise_ratio
        coarse_psnr = peak_signal_noise_ratio(frame, coarse, data_range=255)
        
        # Stage 2: Residual encoding
        # Compute residual
        residual = frame.astype(np.float32) - coarse.astype(np.float32)
        residual = residual / 127.5  # Normalize to [-2, 2] range
        
        # Convert to tensor (B, C, H, W)
        residual_tensor = torch.from_numpy(residual).permute(2, 0, 1).unsqueeze(0).float()
        
        # Encode residuals
        with torch.no_grad():
            compressed_residuals = self.residual_encoder(residual_tensor)
        
        return {
            'pvc_data': {
                'func_ids': func_ids,
                'params': params
            },
            'residual_data': compressed_residuals,
            'coarse_psnr': coarse_psnr
        }
    
    def decode(self, encoded_data):
        """
        Decode frame using hybrid approach.
        
        Args:
            encoded_data: Dict from encode()
        
        Returns:
            Reconstructed frame (H, W, 3) uint8 [0, 255]
        """
        # Stage 1: Procedural decoding
        pvc_data = encoded_data['pvc_data']
        
        # Reconstruct coarse frame (we need a dummy frame for shape)
        # In practice, store frame dimensions in encoded_data
        dummy_frame = np.zeros((256, 256, 3), dtype=np.uint8)
        
        sparse_func_ids = [CONTIGUOUS_TO_SPARSE.get(fid, 0) for fid in pvc_data['func_ids']]
        coarse = np.zeros_like(dummy_frame)
        for fid, param in zip(sparse_func_ids, pvc_data['params']):
            coarse = self.execute_function_top10(fid, param, coarse)
        
        # Stage 2: Residual decoding
        with torch.no_grad():
            residual_tensor = self.residual_decoder(encoded_data['residual_data'])
        
        # Convert residual back to numpy
        residual = residual_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        residual = residual * 127.5  # Denormalize
        
        # Combine
        final = coarse.astype(np.float32) + residual
        final = np.clip(final, 0, 255).astype(np.uint8)
        
        return final
    
    def get_compression_stats(self, encoded_data):
        """
        Get compression statistics.
        
        Args:
            encoded_data: Dict from encode()
        
        Returns:
            Dict with compression stats
        """
        # PVC size (estimate)
        pvc_func_ids = encoded_data['pvc_data']['func_ids']
        pvc_params = encoded_data['pvc_data']['params']
        
        # Assume 1 byte per function ID, 40 bytes per param set (10 floats × 4 bytes)
        pvc_size = len(pvc_func_ids) * 1 + len(pvc_params) * 40
        
        # Residual size
        residual_size = self.residual_encoder.get_compressed_size(encoded_data['residual_data'])
        
        # Original size (256×256×3 = 196608 bytes)
        original_size = 256 * 256 * 3
        
        total_size = pvc_size + residual_size
        compression_ratio = (1 - total_size / original_size) * 100
        
        return {
            'original_size': original_size,
            'pvc_size': pvc_size,
            'residual_size': residual_size,
            'total_size': total_size,
            'compression_ratio': compression_ratio,
            'coarse_psnr': encoded_data['coarse_psnr']
        }


if __name__ == "__main__":
    print("="*70)
    print("Testing Hybrid PVC v2.0 Codec")
    print("="*70)
    
    # Create hybrid codec
    print("\n📦 Initializing hybrid codec...")
    codec = HybridPVCCodec(
        pvc_model_path="/tmp/pvc_v2_perceptual_best.pth",
        residual_quality=20
    )
    print("✅ Codec initialized")
    
    # Generate test frame
    print("\n🎨 Generating test frame...")
    from training.synthetic_generator_extended import ExtendedSyntheticGenerator
    generator = ExtendedSyntheticGenerator(width=256, height=256)
    frames, _ = generator.generate_dataset(num_samples=1, min_functions=5, max_functions=10)
    test_frame = frames[0]
    print(f"✅ Test frame generated: {test_frame.shape}")
    
    # Encode
    print("\n🔄 Encoding...")
    encoded = codec.encode(test_frame)
    stats = codec.get_compression_stats(encoded)
    
    print(f"\n📊 Compression Statistics:")
    print(f"   Original size: {stats['original_size']} bytes ({stats['original_size'] / 1024:.1f} KB)")
    print(f"   PVC size: {stats['pvc_size']} bytes ({stats['pvc_size'] / 1024:.1f} KB)")
    print(f"   Residual size: {stats['residual_size']} bytes ({stats['residual_size'] / 1024:.1f} KB)")
    print(f"   Total size: {stats['total_size']} bytes ({stats['total_size'] / 1024:.1f} KB)")
    print(f"   Compression ratio: {stats['compression_ratio']:.1f}%")
    print(f"   Coarse PSNR: {stats['coarse_psnr']:.2f} dB")
    
    # Decode
    print("\n🔄 Decoding...")
    decoded = codec.decode(encoded)
    print(f"✅ Decoded frame: {decoded.shape}")
    
    # Measure final quality
    print("\n📈 Final Quality:")
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    
    final_psnr = peak_signal_noise_ratio(test_frame, decoded, data_range=255)
    final_ssim = structural_similarity(test_frame, decoded, channel_axis=2, data_range=255)
    
    print(f"   Final PSNR: {final_psnr:.2f} dB")
    print(f"   Final SSIM: {final_ssim:.4f}")
    print(f"   Improvement over coarse: {final_psnr - stats['coarse_psnr']:.2f} dB")
    
    # Save comparison
    print("\n💾 Saving comparison...")
    comparison = np.hstack([test_frame, decoded])
    cv2.imwrite('/tmp/hybrid_pvc_comparison.png', comparison)
    print("   Saved: /tmp/hybrid_pvc_comparison.png")
    
    print("\n" + "="*70)
    print("✅ Hybrid Codec Test Complete!")
    print("="*70)
    
    # Summary
    if final_psnr >= 30:
        print("\n🎉 EXCELLENT! Achieved 30+ dB target!")
    elif final_psnr >= 20:
        print("\n✅ GOOD! Approaching target (20-30 dB)")
    else:
        print("\n⚠️  Needs training to reach target")

