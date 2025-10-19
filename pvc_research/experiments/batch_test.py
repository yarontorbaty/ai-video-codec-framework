#!/usr/bin/env python3
"""
PVC Batch Test Script

Tests multiple anime clips and generates comparison report.
"""

import sys
import os
from pathlib import Path
import json
import subprocess
import time
from datetime import datetime
import logging

# Add pvc_research to path
pvc_root = Path(__file__).parent.parent
sys.path.insert(0, str(pvc_root))

# Import PVC modules directly
sys.path.insert(0, str(pvc_root))
from encoder.contour_extractor import ContourExtractor
from encoder.motion_tracker import MotionTracker
from decoder.procedural_textures import ProceduralTextures
from decoder.scene_renderer import SceneRenderer
from utils.quality_metrics import QualityMetrics
from utils.bitrate_calculator import BitrateCalculator

# Import main encoder/decoder classes
import importlib.util
encoder_spec = importlib.util.spec_from_file_location("pvc_encoder", pvc_root / "encoder.py")
encoder_module = importlib.util.module_from_spec(encoder_spec)
encoder_spec.loader.exec_module(encoder_module)
PVCEncoder = encoder_module.PVCEncoder

decoder_spec = importlib.util.spec_from_file_location("pvc_decoder", pvc_root / "decoder.py")
decoder_module = importlib.util.module_from_spec(decoder_spec)
decoder_spec.loader.exec_module(decoder_module)
PVCDecoder = decoder_module.PVCDecoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_video_info(video_path):
    """Get video metadata using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)
    except:
        return None


def create_av1_baseline(input_path, output_path, target_bitrate='5M'):
    """Create AV1 baseline encode for comparison"""
    logger.info(f"Creating AV1 baseline: {output_path}")
    
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-c:v', 'libaom-av1',
        '-b:v', target_bitrate,
        '-cpu-used', '4',  # Speed preset (0=slowest, 8=fastest)
        '-an',  # No audio
        output_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        logger.info(f"✅ Created AV1 baseline")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ AV1 encoding failed: {e}")
        return False


def test_single_clip(clip_path, output_dir, baseline_dir):
    """Test a single anime clip"""
    clip_name = Path(clip_path).stem
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {clip_name}")
    logger.info(f"{'='*60}")
    
    result = {
        'clip_name': clip_name,
        'input_path': str(clip_path),
        'timestamp': datetime.now().isoformat()
    }
    
    # Get input video info
    video_info = get_video_info(clip_path)
    if video_info:
        format_info = video_info.get('format', {})
        result['input_size_bytes'] = int(format_info.get('size', 0))
        result['input_duration_sec'] = float(format_info.get('duration', 0))
        result['input_bitrate_bps'] = int(format_info.get('bit_rate', 0))
    
    # Define output paths
    scene_json = output_dir / f"{clip_name}_scene.json"
    reconstructed_video = output_dir / f"{clip_name}_reconstructed.mp4"
    av1_baseline = baseline_dir / f"{clip_name}_av1.mp4"
    
    try:
        # 1. Encode with PVC
        logger.info("\n📦 Encoding with PVC...")
        start_time = time.time()
        
        encoder = PVCEncoder()
        scene_desc = encoder.encode_video(str(clip_path))
        encoder.save_scene(scene_desc, str(scene_json))
        
        encoding_time = time.time() - start_time
        result['encoding_time_sec'] = encoding_time
        
        # Get scene size
        scene_size = Path(scene_json).stat().st_size
        result['scene_size_bytes'] = scene_size
        
        # Calculate PVC bitrate
        duration = scene_desc['metadata']['duration']
        bitrate_info = BitrateCalculator.calculate_bitrate(scene_desc, duration)
        result['pvc_bitrate_mbps'] = bitrate_info['bitrate_mbps']
        
        logger.info(f"  Scene size: {scene_size / 1024:.2f} KB")
        logger.info(f"  Encoding time: {encoding_time:.1f}s")
        logger.info(f"  Bitrate: {bitrate_info['bitrate_mbps']:.3f} Mbps")
        
        # 2. Decode with PVC
        logger.info("\n📤 Decoding with PVC...")
        start_time = time.time()
        
        decoder = PVCDecoder()
        decoder.decode_scene(str(scene_json), str(reconstructed_video))
        
        decoding_time = time.time() - start_time
        result['decoding_time_sec'] = decoding_time
        
        logger.info(f"  Decoding time: {decoding_time:.1f}s")
        
        # 3. Create AV1 baseline
        logger.info("\n🎬 Creating AV1 baseline...")
        baseline_created = create_av1_baseline(str(clip_path), str(av1_baseline))
        
        if baseline_created:
            av1_size = Path(av1_baseline).stat().st_size
            result['av1_size_bytes'] = av1_size
            
            av1_info = get_video_info(str(av1_baseline))
            if av1_info:
                av1_format = av1_info.get('format', {})
                result['av1_bitrate_bps'] = int(av1_format.get('bit_rate', 0))
        
        # 4. Calculate quality metrics
        logger.info("\n📊 Calculating quality metrics...")
        metrics = QualityMetrics.compare_videos(
            str(clip_path),
            str(reconstructed_video),
            use_vmaf=False
        )
        
        result['quality_metrics'] = {
            'psnr_db': metrics['avg_psnr_db'],
            'ssim': metrics['avg_ssim'],
            'min_psnr_db': metrics['min_psnr_db'],
            'max_psnr_db': metrics['max_psnr_db']
        }
        
        logger.info(f"  PSNR: {metrics['avg_psnr_db']:.2f} dB")
        logger.info(f"  SSIM: {metrics['avg_ssim']:.4f}")
        
        # 5. Calculate compression ratios
        logger.info("\n💾 Compression analysis...")
        
        # PVC vs original
        pvc_reduction = (1 - scene_size / result['input_size_bytes']) * 100
        result['pvc_vs_original_reduction_percent'] = pvc_reduction
        
        # PVC vs AV1
        if baseline_created:
            av1_reduction = (1 - scene_size / av1_size) * 100
            result['pvc_vs_av1_reduction_percent'] = av1_reduction
            
            logger.info(f"  Original: {result['input_size_bytes'] / (1024*1024):.2f} MB")
            logger.info(f"  AV1: {av1_size / (1024*1024):.2f} MB")
            logger.info(f"  PVC: {scene_size / 1024:.2f} KB")
            logger.info(f"  PVC vs AV1: {av1_reduction:.1f}% reduction")
        else:
            logger.info(f"  Original: {result['input_size_bytes'] / (1024*1024):.2f} MB")
            logger.info(f"  PVC: {scene_size / 1024:.2f} KB")
            logger.info(f"  PVC vs Original: {pvc_reduction:.1f}% reduction")
        
        result['status'] = 'success'
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        result['status'] = 'failed'
        result['error'] = str(e)
    
    return result


def generate_report(results, output_path):
    """Generate summary report"""
    logger.info(f"\n{'='*60}")
    logger.info("BATCH TEST SUMMARY")
    logger.info(f"{'='*60}\n")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    logger.info(f"Total clips tested: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}\n")
    
    if successful:
        logger.info("Results by clip:")
        logger.info("-" * 60)
        
        for r in successful:
            logger.info(f"\n{r['clip_name']}:")
            logger.info(f"  Quality: PSNR={r['quality_metrics']['psnr_db']:.2f}dB, SSIM={r['quality_metrics']['ssim']:.4f}")
            logger.info(f"  Compression: {r.get('pvc_vs_av1_reduction_percent', r['pvc_vs_original_reduction_percent']):.1f}% reduction")
            logger.info(f"  Encoding: {r['encoding_time_sec']:.1f}s, Decoding: {r['decoding_time_sec']:.1f}s")
        
        # Calculate averages
        avg_psnr = sum(r['quality_metrics']['psnr_db'] for r in successful) / len(successful)
        avg_ssim = sum(r['quality_metrics']['ssim'] for r in successful) / len(successful)
        avg_compression = sum(r.get('pvc_vs_av1_reduction_percent', r['pvc_vs_original_reduction_percent']) for r in successful) / len(successful)
        
        logger.info(f"\n{'='*60}")
        logger.info("AVERAGES:")
        logger.info(f"  PSNR: {avg_psnr:.2f} dB")
        logger.info(f"  SSIM: {avg_ssim:.4f}")
        logger.info(f"  Compression: {avg_compression:.1f}%")
        logger.info(f"{'='*60}\n")
        
        # Check if target achieved
        target_met = avg_compression >= 90.0
        logger.info(f"🎯 Target (90% reduction): {'✅ ACHIEVED' if target_met else '❌ NOT MET'}\n")
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_clips': len(results),
            'successful': len(successful),
            'failed': len(failed)
        },
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📄 Report saved: {output_path}")


def main():
    """Run batch tests on all clips in test_clips directory"""
    # Setup paths
    base_dir = Path(__file__).parent.parent
    test_clips_dir = base_dir / 'test_clips'
    output_dir = base_dir / 'experiments' / 'results'
    baseline_dir = base_dir / 'experiments' / 'baselines'
    report_path = base_dir / 'experiments' / 'report.json'
    
    # Find all video files
    video_extensions = ['.mp4', '.mkv', '.mov', '.avi']
    clips = []
    for ext in video_extensions:
        clips.extend(list(test_clips_dir.glob(f'*{ext}')))
    
    # Filter out README
    clips = [c for c in clips if c.stem != 'README']
    
    if not clips:
        logger.error("❌ No video clips found in test_clips/")
        logger.info("   Drop your 3 anime clips in: pvc_research/test_clips/")
        sys.exit(1)
    
    logger.info(f"Found {len(clips)} clip(s) to test:")
    for clip in clips:
        logger.info(f"  - {clip.name}")
    
    # Test each clip
    results = []
    for clip in clips:
        result = test_single_clip(clip, output_dir, baseline_dir)
        results.append(result)
    
    # Generate report
    generate_report(results, report_path)
    
    logger.info("\n✅ Batch test complete!")
    logger.info(f"   Results: {output_dir}")
    logger.info(f"   Report: {report_path}")


if __name__ == '__main__':
    main()

