#!/usr/bin/env python3
"""
PVC Decoder - Main Entry Point

Decodes procedural scene description (ISP) back into video.

Usage:
    python decoder.py --input scene.json --output reconstructed.mp4 [options]
"""

import argparse
import json
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from decoder.scene_renderer import SceneRenderer
from utils.quality_metrics import QualityMetrics
from utils.bitrate_calculator import BitrateCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PVCDecoder:
    """
    Main decoder class for Procedural Video Codec.
    """
    
    def __init__(self):
        """Initialize decoder."""
        self.renderer = None
    
    def decode_scene(self, scene_path: str, output_path: str):
        """
        Decode a scene description back into video.
        
        Args:
            scene_path: Path to scene JSON file
            output_path: Output video path
        """
        logger.info(f"📂 Loading scene: {scene_path}")
        
        # Load scene description
        with open(scene_path, 'r') as f:
            scene_desc = json.load(f)
        
        # Get metadata
        metadata = scene_desc.get('metadata', {})
        resolution = metadata.get('resolution', [1280, 720])
        fps = metadata.get('fps', 30)
        frame_count = metadata.get('frame_count', 30)
        
        logger.info(f"  Resolution: {resolution[0]}x{resolution[1]}")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Frames: {frame_count}")
        
        # Initialize renderer
        self.renderer = SceneRenderer(resolution=tuple(resolution))
        
        # Render frames
        logger.info("🎨 Rendering frames...")
        frames = self.renderer.render_sequence(scene_desc, num_frames=frame_count)
        
        # Save video
        logger.info(f"💾 Saving video: {output_path}")
        self.renderer.save_video(frames, output_path, fps=int(fps))
        
        logger.info("✅ Decoding complete!")
    
    def decode_and_evaluate(self,
                          scene_path: str,
                          output_path: str,
                          original_path: str = None):
        """
        Decode scene and optionally evaluate quality.
        
        Args:
            scene_path: Path to scene JSON
            output_path: Output video path
            original_path: Optional original video for comparison
        """
        # Decode
        self.decode_scene(scene_path, output_path)
        
        # Evaluate if original provided
        if original_path:
            logger.info("📊 Evaluating quality...")
            
            metrics = QualityMetrics.compare_videos(
                original_path,
                output_path,
                use_vmaf=False  # Set to True if FFmpeg has VMAF support
            )
            
            logger.info(f"  PSNR: {metrics['avg_psnr_db']:.2f} dB")
            logger.info(f"  SSIM: {metrics['avg_ssim']:.4f}")
            
            if 'vmaf' in metrics and metrics['vmaf'] is not None:
                logger.info(f"  VMAF: {metrics['vmaf']:.2f}")
            
            # Calculate compression ratio
            logger.info("📊 Compression analysis...")
            
            with open(scene_path, 'r') as f:
                scene_desc = json.load(f)
            
            duration = scene_desc['metadata'].get('duration', 1.0)
            bitrate_info = BitrateCalculator.calculate_bitrate(scene_desc, duration)
            
            # Estimate original bitrate (assume high quality)
            # For anime/animation, typical bitrate is 3-8 Mbps
            estimated_original_bitrate = 5.0  # Mbps
            
            comparison = BitrateCalculator.compare_to_baseline(
                pvc_bitrate=bitrate_info['bitrate_mbps'],
                baseline_bitrate=estimated_original_bitrate
            )
            
            logger.info(f"  PVC bitrate: {comparison['pvc_mbps']:.3f} Mbps")
            logger.info(f"  Baseline: {comparison['baseline_mbps']:.3f} Mbps")
            logger.info(f"  Reduction: {comparison['reduction_percent']:.1f}%")
            logger.info(f"  Target (90%) achieved: {'✅' if comparison['achieved_target'] else '❌'}")


def main():
    parser = argparse.ArgumentParser(
        description='Procedural Video Codec (PVC) Decoder',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input scene JSON file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output video file')
    parser.add_argument('--original',
                       help='Original video for quality evaluation')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate quality metrics')
    
    args = parser.parse_args()
    
    # Decode
    decoder = PVCDecoder()
    
    if args.evaluate and args.original:
        decoder.decode_and_evaluate(args.input, args.output, args.original)
    else:
        decoder.decode_scene(args.input, args.output)
    
    logger.info("✅ Done!")


if __name__ == '__main__':
    main()

