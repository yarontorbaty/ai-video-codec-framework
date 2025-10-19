#!/usr/bin/env python3
"""
PVC Quick Test - Verify encoder/decoder pipeline works

Creates a simple test video, encodes it, decodes it, and checks metrics.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import tempfile
import logging

# Add pvc_research to path
sys.path.insert(0, str(Path(__file__).parent / 'pvc_research'))

from encoder.contour_extractor import ContourExtractor
from encoder.motion_tracker import MotionTracker
from decoder.procedural_textures import ProceduralTextures
from decoder.scene_renderer import SceneRenderer
from utils.quality_metrics import QualityMetrics
from utils.bitrate_calculator import BitrateCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_video(output_path: str, num_frames: int = 30):
    """
    Create a simple test animation video.
    
    Animates a moving rectangle and circle - ideal for PVC.
    """
    logger.info(f"🎬 Creating test video: {num_frames} frames")
    
    width, height = 640, 480
    fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        # Create frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Moving rectangle
        x = 100 + i * 5
        cv2.rectangle(frame, (x, 100), (x + 100, 200), (100, 150, 250), -1)
        
        # Moving circle
        y = 300 + int(30 * np.sin(i * 0.2))
        cv2.circle(frame, (300, y), 50, (250, 200, 100), -1)
        
        # Static background shape
        cv2.rectangle(frame, (0, 400), (width, height), (50, 50, 50), -1)
        
        out.write(frame)
    
    out.release()
    logger.info(f"✅ Created test video: {output_path}")


def test_pvc_pipeline():
    """
    Test complete PVC encoder/decoder pipeline.
    """
    logger.info("="*60)
    logger.info("PVC PIPELINE TEST")
    logger.info("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test video
        original_video = tmpdir / "test_original.mp4"
        create_test_video(str(original_video), num_frames=30)
        
        # Encode
        logger.info("\n📦 ENCODING...")
        logger.info("-"*60)
        
        from encoder import PVCEncoder
        encoder = PVCEncoder()
        scene_json = tmpdir / "scene.json"
        
        scene_desc = encoder.encode_video(str(original_video))
        encoder.save_scene(scene_desc, str(scene_json))
        
        # Check encoding results
        sizes = BitrateCalculator.calculate_scene_size(scene_desc)
        duration = scene_desc['metadata']['duration']
        bitrate_info = BitrateCalculator.calculate_bitrate(scene_desc, duration)
        
        logger.info(f"\n📊 Encoding Results:")
        logger.info(f"  Scene size: {sizes['total'] / 1024:.2f} KB")
        logger.info(f"  Contours: {sizes['contours']} bytes")
        logger.info(f"  Motion: {sizes['motion']} bytes")
        logger.info(f"  Textures: {sizes['textures']} bytes")
        logger.info(f"  Bitrate: {bitrate_info['bitrate_mbps']:.3f} Mbps")
        
        # Decode
        logger.info("\n📤 DECODING...")
        logger.info("-"*60)
        
        from decoder import PVCDecoder
        decoder = PVCDecoder()
        reconstructed_video = tmpdir / "reconstructed.mp4"
        
        decoder.decode_scene(str(scene_json), str(reconstructed_video))
        
        # Evaluate quality
        logger.info("\n📊 QUALITY EVALUATION...")
        logger.info("-"*60)
        
        orig_frames = QualityMetrics.load_video_frames(str(original_video))
        recon_frames = QualityMetrics.load_video_frames(str(reconstructed_video))
        
        metrics = QualityMetrics.calculate_video_metrics(orig_frames, recon_frames)
        
        logger.info(f"\n✨ Quality Metrics:")
        logger.info(f"  PSNR: {metrics['avg_psnr_db']:.2f} dB")
        logger.info(f"  SSIM: {metrics['avg_ssim']:.4f}")
        logger.info(f"  Min PSNR: {metrics['min_psnr_db']:.2f} dB")
        logger.info(f"  Max PSNR: {metrics['max_psnr_db']:.2f} dB")
        
        # Compression analysis
        logger.info("\n💾 COMPRESSION ANALYSIS...")
        logger.info("-"*60)
        
        # Estimate original bitrate
        original_size = Path(original_video).stat().st_size
        original_bitrate_mbps = (original_size * 8) / (duration * 1_000_000)
        
        comparison = BitrateCalculator.compare_to_baseline(
            pvc_bitrate=bitrate_info['bitrate_mbps'],
            baseline_bitrate=original_bitrate_mbps
        )
        
        logger.info(f"  Original: {comparison['baseline_mbps']:.3f} Mbps ({original_size / 1024:.1f} KB)")
        logger.info(f"  PVC: {comparison['pvc_mbps']:.3f} Mbps ({sizes['total'] / 1024:.1f} KB)")
        logger.info(f"  Reduction: {comparison['reduction_percent']:.1f}%")
        logger.info(f"  Compression ratio: {comparison['compression_ratio']:.1f}x")
        logger.info(f"  Target (90%) achieved: {'✅ YES' if comparison['achieved_target'] else '❌ NO'}")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        
        success = (
            metrics['avg_psnr_db'] > 25 and
            metrics['avg_ssim'] > 0.7 and
            comparison['reduction_percent'] > 50  # Lower bar for test
        )
        
        if success:
            logger.info("✅ PVC PIPELINE TEST PASSED!")
            logger.info(f"   Quality: PSNR={metrics['avg_psnr_db']:.1f}dB, SSIM={metrics['avg_ssim']:.3f}")
            logger.info(f"   Compression: {comparison['reduction_percent']:.1f}% reduction")
        else:
            logger.warning("⚠️  PVC PIPELINE TEST: Results below expectations")
            logger.warning(f"   Quality: PSNR={metrics['avg_psnr_db']:.1f}dB (target >25)")
            logger.warning(f"   SSIM: {metrics['avg_ssim']:.3f} (target >0.7)")
            logger.warning(f"   Compression: {comparison['reduction_percent']:.1f}% (target >50%)")
        
        logger.info("\n💡 Next steps:")
        logger.info("   1. Test with real anime clip")
        logger.info("   2. Deploy to AWS worker")
        logger.info("   3. Integrate with dashboard")
        logger.info("="*60)
        
        return success


if __name__ == '__main__':
    try:
        success = test_pvc_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ TEST FAILED: {e}", exc_info=True)
        sys.exit(1)

