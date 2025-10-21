#!/usr/bin/env python3
"""
PVC Complete Pipeline - Encoder + Decoder with Residuals

Full pipeline that produces near-perfect reconstructions.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import json
import logging
from typing import Dict, List

# Add pvc_research to path
pvc_root = Path(__file__).parent
sys.path.insert(0, str(pvc_root))

# Import modules
import importlib.util

# Load encoder
encoder_spec = importlib.util.spec_from_file_location("pvc_encoder", pvc_root / "encoder.py")
encoder_module = importlib.util.module_from_spec(encoder_spec)
encoder_spec.loader.exec_module(encoder_module)
PVCEncoder = encoder_module.PVCEncoder

# Load decoder
decoder_spec = importlib.util.spec_from_file_location("pvc_decoder", pvc_root / "decoder.py")
decoder_module = importlib.util.module_from_spec(decoder_spec)
decoder_spec.loader.exec_module(decoder_module)
PVCDecoder = decoder_module.PVCDecoder

# Load residual encoder
from encoder.residual_encoder import ResidualEncoder
from utils.quality_metrics import QualityMetrics
from utils.bitrate_calculator import BitrateCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PVCCompletePipeline:
    """
    Complete PVC pipeline with residuals.
    
    Pipeline:
    1. Encode video geometrically → scene.json
    2. Decode scene.json → reconstructed video (geometric)
    3. Compute residuals → high-error tiles
    4. Add residuals to scene
    5. Decode with residuals → final video (near-perfect)
    """
    
    def __init__(self, error_threshold: float = 10.0, tile_size: int = 64):
        """
        Initialize complete pipeline.
        
        Args:
            error_threshold: MSE threshold for residual tiles
            tile_size: Residual tile size
        """
        self.encoder = PVCEncoder()
        self.decoder = PVCDecoder()
        self.residual_encoder = ResidualEncoder(
            error_threshold=error_threshold,
            tile_size=tile_size
        )
    
    def encode_with_residuals(self,
                            input_video: str,
                            output_scene: str,
                            temp_reconstructed: str = None) -> Dict:
        """
        Complete encoding pipeline with residuals.
        
        Args:
            input_video: Input video path
            output_scene: Output scene JSON path
            temp_reconstructed: Temp file for geometric reconstruction
            
        Returns:
            Statistics dictionary
        """
        if temp_reconstructed is None:
            temp_reconstructed = output_scene.replace('.json', '_temp_recon.mp4')
        
        logger.info("="*60)
        logger.info("PVC COMPLETE ENCODING PIPELINE")
        logger.info("="*60)
        
        # Step 1: Encode geometrically
        logger.info("\n📦 STEP 1: Geometric Encoding")
        logger.info("-"*60)
        scene_desc = self.encoder.encode_video(input_video)
        
        # Get scene size (before residuals)
        geometric_size = BitrateCalculator.calculate_scene_size(scene_desc)
        logger.info(f"Geometric scene size: {geometric_size['total'] / 1024:.2f} KB")
        
        # Step 2: Decode geometrically (to get reconstruction for residuals)
        logger.info("\n📤 STEP 2: Geometric Decoding (for residual computation)")
        logger.info("-"*60)
        
        # Save temp scene (without residuals)
        temp_scene = output_scene.replace('.json', '_temp_scene.json')
        self.encoder.save_scene(scene_desc, temp_scene)
        
        # Decode
        self.decoder.decode_scene(temp_scene, temp_reconstructed)
        
        # Step 3: Load frames for residual computation
        logger.info("\n🔍 STEP 3: Computing Residuals")
        logger.info("-"*60)
        
        original_frames = QualityMetrics.load_video_frames(input_video)
        reconstructed_frames = QualityMetrics.load_video_frames(temp_reconstructed)
        
        # Compute residuals
        residuals = self.residual_encoder.compute_residuals(
            original_frames,
            reconstructed_frames
        )
        
        # Add residuals to scene
        scene_desc['residuals'] = residuals
        
        # Step 4: Calculate sizes
        logger.info("\n📊 STEP 4: Size Analysis")
        logger.info("-"*60)
        
        residual_size = self.residual_encoder.estimate_residual_size(residuals)
        
        # Save complete scene (with residuals)
        self.encoder.save_scene(scene_desc, output_scene)
        
        # Get total size
        import os
        total_size_bytes = os.path.getsize(output_scene)
        
        logger.info(f"Geometric: {geometric_size['total'] / 1024:.2f} KB")
        logger.info(f"Residuals: {residual_size['total_kb']:.2f} KB ({residual_size['total_tiles']} tiles)")
        logger.info(f"Total: {total_size_bytes / 1024:.2f} KB")
        
        # Calculate compression
        original_size = os.path.getsize(input_video)
        compression_ratio = (1 - total_size_bytes / original_size) * 100
        
        logger.info(f"\nOriginal video: {original_size / (1024*1024):.2f} MB")
        logger.info(f"PVC (with residuals): {total_size_bytes / 1024:.2f} KB")
        logger.info(f"Compression: {compression_ratio:.1f}%")
        
        return {
            'geometric_size_kb': geometric_size['total'] / 1024,
            'residual_size_kb': residual_size['total_kb'],
            'total_size_kb': total_size_bytes / 1024,
            'residual_tiles': residual_size['total_tiles'],
            'original_size_mb': original_size / (1024*1024),
            'compression_percent': compression_ratio
        }
    
    def decode_with_residuals(self,
                            input_scene: str,
                            output_video: str) -> Dict:
        """
        Decode scene with residuals.
        
        Args:
            input_scene: Scene JSON path (with residuals)
            output_video: Output video path
            
        Returns:
            Statistics dictionary
        """
        logger.info("\n📂 Loading scene with residuals...")
        
        with open(input_scene, 'r') as f:
            scene_desc = json.load(f)
        
        metadata = scene_desc.get('metadata', {})
        resolution = metadata.get('resolution', [1920, 1080])
        fps = metadata.get('fps', 30)
        frame_count = metadata.get('frame_count', 30)
        
        logger.info(f"  Resolution: {resolution[0]}x{resolution[1]}")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Frames: {frame_count}")
        
        residuals = scene_desc.get('residuals', [])
        logger.info(f"  Residuals: {len(residuals)} frames")
        
        # Step 1: Render geometrically
        logger.info("\n🎨 Rendering geometric base...")
        self.decoder.renderer = None  # Reset renderer
        frames = self.decoder.renderer.render_sequence(scene_desc, num_frames=frame_count) if self.decoder.renderer else []
        
        if not frames:
            from decoder.scene_renderer import SceneRenderer
            renderer = SceneRenderer(resolution=tuple(resolution))
            frames = renderer.render_sequence(scene_desc, num_frames=frame_count)
        
        # Step 2: Apply residuals
        if residuals:
            logger.info("\n🔧 Applying residuals...")
            frames = self._apply_residuals_to_frames(frames, residuals)
        else:
            logger.info("\n⚠️  No residuals found, using geometric rendering only")
        
        # Step 3: Save video
        logger.info(f"\n💾 Saving final video: {output_video}")
        from decoder.scene_renderer import SceneRenderer
        renderer = SceneRenderer(resolution=tuple(resolution))
        renderer.save_video(frames, output_video, fps=int(fps))
        
        logger.info("✅ Decoding complete!")
        
        return {
            'frame_count': len(frames),
            'resolution': resolution,
            'residuals_applied': len(residuals) > 0,
            'residual_tiles': sum(len(r.get('tiles', [])) for r in residuals)
        }
    
    def _apply_residuals_to_frames(self,
                                   frames: List[np.ndarray],
                                   residuals: List[Dict]) -> List[np.ndarray]:
        """
        Apply residual corrections to frames.
        
        Args:
            frames: List of geometrically reconstructed frames
            residuals: List of residual data per frame
            
        Returns:
            Corrected frames
        """
        corrected_frames = frames.copy()
        
        tiles_applied = 0
        for frame_res in residuals:
            frame_idx = frame_res.get('frame_idx', 0)
            tiles = frame_res.get('tiles', [])
            
            if frame_idx >= len(corrected_frames):
                continue
            
            frame = corrected_frames[frame_idx]
            
            for tile in tiles:
                position = tile.get('position', [0, 0])
                size = tile.get('size', [64, 64])
                data = tile.get('data', None)
                
                if data is None:
                    continue
                
                # Decode base64 to bytes
                import base64
                tile_bytes = base64.b64decode(data)
                
                # Decode tile (PNG compressed)
                tile_img = cv2.imdecode(
                    np.frombuffer(tile_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )
                
                if tile_img is None:
                    continue
                
                # Apply tile to frame
                x, y = position
                h, w = size
                
                # Clip to frame bounds
                x2 = min(x + w, frame.shape[1])
                y2 = min(y + h, frame.shape[0])
                
                if x2 > x and y2 > y:
                    # Resize tile if needed
                    if tile_img.shape[0] != (y2-y) or tile_img.shape[1] != (x2-x):
                        tile_img = cv2.resize(tile_img, (x2-x, y2-y))
                    
                    # Replace region with residual tile
                    frame[y:y2, x:x2] = tile_img
                    tiles_applied += 1
            
            corrected_frames[frame_idx] = frame
        
        logger.info(f"  Applied {tiles_applied} residual tiles")
        
        return corrected_frames


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PVC Complete Pipeline with Residuals'
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input video file')
    parser.add_argument('--output-scene', '-s', required=True,
                       help='Output scene JSON file')
    parser.add_argument('--output-video', '-o', required=True,
                       help='Output reconstructed video file')
    parser.add_argument('--error-threshold', type=float, default=10.0,
                       help='MSE threshold for residuals (default: 10.0)')
    parser.add_argument('--tile-size', type=int, default=64,
                       help='Residual tile size (default: 64)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Calculate quality metrics')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = PVCCompletePipeline(
        error_threshold=args.error_threshold,
        tile_size=args.tile_size
    )
    
    # Encode
    encode_stats = pipeline.encode_with_residuals(
        args.input,
        args.output_scene
    )
    
    # Decode
    decode_stats = pipeline.decode_with_residuals(
        args.output_scene,
        args.output_video
    )
    
    # Evaluate
    if args.evaluate:
        logger.info("\n📊 QUALITY EVALUATION")
        logger.info("="*60)
        
        metrics = QualityMetrics.compare_videos(
            args.input,
            args.output_video,
            use_vmaf=False
        )
        
        logger.info(f"PSNR: {metrics['avg_psnr_db']:.2f} dB")
        logger.info(f"SSIM: {metrics['avg_ssim']:.4f}")
        
        logger.info("\n🎯 FINAL RESULTS")
        logger.info("="*60)
        logger.info(f"Compression: {encode_stats['compression_percent']:.1f}%")
        logger.info(f"Quality: PSNR={metrics['avg_psnr_db']:.1f}dB, SSIM={metrics['avg_ssim']:.3f}")
        logger.info(f"Size: {encode_stats['total_size_kb']:.1f} KB ({encode_stats['residual_tiles']} residual tiles)")
    
    logger.info("\n✅ Pipeline complete!")


if __name__ == '__main__':
    main()

