#!/usr/bin/env python3
"""
PVC Encoder - Main Entry Point

Encodes video into procedural scene description (ISP).

Usage:
    python encoder.py --input video.mp4 --output scene.json [options]
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
import logging
import sys
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from encoder.contour_extractor import ContourExtractor
from encoder.motion_tracker import MotionTracker
from utils.bitrate_calculator import BitrateCalculator
from utils.quality_metrics import QualityMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PVCEncoder:
    """
    Main encoder class for Procedural Video Codec.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize encoder with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.contour_extractor = ContourExtractor(
            canny_low=self.config.get('canny_low', 50),
            canny_high=self.config.get('canny_high', 150),
            min_contour_area=self.config.get('min_contour_area', 100)
        )
        
        self.motion_tracker = MotionTracker(
            flow_method=self.config.get('flow_method', 'farneback'),
            keyframe_interval=self.config.get('keyframe_interval', 30)
        )
    
    def encode_video(self, video_path: str, max_frames: int = None) -> Dict:
        """
        Encode a video into procedural scene description.
        
        Args:
            video_path: Path to input video
            max_frames: Maximum frames to process (None = all)
            
        Returns:
            Scene description dictionary (ISP)
        """
        logger.info(f"🎬 Encoding video: {video_path}")
        
        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            frame_count = min(frame_count, max_frames)
        
        duration = frame_count / fps if fps > 0 else 1.0
        
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Frames: {frame_count}")
        logger.info(f"  Duration: {duration:.2f}s")
        
        # Load frames
        frames = []
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
            if (i + 1) % 30 == 0:
                logger.info(f"  Loaded {i + 1}/{frame_count} frames")
        
        cap.release()
        
        # Extract contours from all frames
        logger.info("🔍 Extracting contours...")
        frame_contours = []
        for i, frame in enumerate(frames):
            contours = self.contour_extractor.extract_frame(frame)
            frame_contours.append(contours)
            
            if (i + 1) % 30 == 0:
                logger.info(f"  Processed {i + 1}/{len(frames)} frames")
        
        # Track motion across frames
        logger.info("🎯 Tracking motion...")
        tracked_objects = self.motion_tracker.track_objects(frames, frame_contours)
        
        # Fit motion models
        logger.info("📐 Fitting motion models...")
        for obj in tracked_objects:
            motion_seq = obj.get('motion_sequence', [])
            if motion_seq:
                obj['motion_model'] = self.motion_tracker.fit_motion_model(motion_seq)
        
        # Analyze textures (extract real colors from video frames)
        logger.info("🎨 Analyzing textures...")
        self._assign_textures(tracked_objects, frames)
        
        # Build scene description
        scene_desc = {
            'metadata': {
                'resolution': [width, height],
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'encoder_version': '1.0',
                'encoding_params': self.config
            },
            'objects': tracked_objects,
            'residuals': []  # TODO: Add residual encoding
        }
        
        logger.info("✅ Encoding complete!")
        
        # Calculate size
        sizes = BitrateCalculator.calculate_scene_size(scene_desc)
        bitrate_info = BitrateCalculator.calculate_bitrate(scene_desc, duration)
        
        logger.info(f"📊 Scene size: {sizes['total'] / 1024:.2f} KB")
        logger.info(f"📊 Bitrate: {bitrate_info['bitrate_mbps']:.3f} Mbps")
        
        return scene_desc
    
    def _assign_textures(self, objects: List[Dict], frames: List[np.ndarray] = None):
        """
        Assign textures to objects by extracting real colors AND texture patches.
        
        Args:
            objects: List of tracked objects with contours
            frames: List of video frames to sample colors and textures from
        """
        import base64
        
        for obj in objects:
            if not obj.get('contour_sequence') or not frames:
                continue
            
            # Sample texture from first frame where object appears
            first_frame_idx = obj.get('keyframes', [0])[0]
            if first_frame_idx >= len(frames):
                first_frame_idx = 0
            
            frame = frames[first_frame_idx]
            contour = obj['contour_sequence'][0] if obj['contour_sequence'] else None
            
            if contour is None:
                continue
            
            # Extract average color from object region
            avg_color = self._extract_region_color(frame, contour)
            
            # Extract texture complexity (variance)
            texture_variance = self._extract_texture_variance(frame, contour)
            
            # Get object area to decide texture patch size
            area = contour.get('area', 0)
            
            # Assign texture based on complexity and size
            if texture_variance < 50:  # Very low variance = solid color
                obj['texture'] = {
                    'type': 'solid',
                    'color': avg_color.tolist(),
                    'params': {}
                }
            elif area < 500:  # Small object = just color
                obj['texture'] = {
                    'type': 'solid',
                    'color': avg_color.tolist(),
                    'params': {}
                }
            else:  # High variance or large object = extract texture patch
                # Extract texture patch from object region
                texture_patch = self._extract_texture_patch(frame, contour, patch_size=16)
                
                if texture_patch is not None:
                    # Compress patch with JPEG
                    _, encoded = cv2.imencode('.jpg', texture_patch, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    patch_b64 = base64.b64encode(encoded.tobytes()).decode('ascii')
                    
                    obj['texture'] = {
                        'type': 'patch',
                        'color': avg_color.tolist(),  # Fallback color
                        'patch': patch_b64,
                        'patch_size': list(texture_patch.shape[:2]),  # [height, width]
                        'variance': float(texture_variance),
                        'params': {}
                    }
                else:
                    # Fallback to solid color
                    obj['texture'] = {
                        'type': 'solid',
                        'color': avg_color.tolist(),
                        'params': {}
                    }
    
    def _extract_region_color(self, frame: np.ndarray, contour: Dict) -> np.ndarray:
        """Extract average color from a contour region."""
        # Get bounding box
        x, y, w, h = contour['bounding_box']
        
        # Clip to frame bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        
        if x2 <= x1 or y2 <= y1:
            return np.array([0.5, 0.5, 0.5])  # Gray default
        
        # Extract region
        region = frame[y1:y2, x1:x2]
        
        # Calculate average color (in BGR, convert to RGB for consistency)
        avg_bgr = np.mean(region, axis=(0, 1))
        avg_rgb = avg_bgr[[2, 1, 0]]  # BGR to RGB
        
        # Normalize to [0, 1]
        return avg_rgb / 255.0
    
    def _extract_texture_variance(self, frame: np.ndarray, contour: Dict) -> float:
        """Calculate texture complexity (variance) in a region."""
        # Get bounding box
        x, y, w, h = contour['bounding_box']
        
        # Clip to frame bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Extract region and convert to grayscale
        region = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Calculate variance as measure of texture complexity
        return float(np.var(gray))
    
    def _extract_texture_patch(self, frame: np.ndarray, contour: Dict, patch_size: int = 16) -> np.ndarray:
        """
        Extract a representative texture patch from an object region.
        
        Args:
            frame: Video frame
            contour: Object contour
            patch_size: Size of square patch to extract
            
        Returns:
            Texture patch (patch_size x patch_size x 3) or None
        """
        # Get bounding box
        x, y, w, h = contour['bounding_box']
        
        # Clip to frame bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract region
        region = frame[y1:y2, x1:x2]
        
        # If region is smaller than patch_size, use entire region
        if region.shape[0] < patch_size or region.shape[1] < patch_size:
            # Resize to patch_size
            if region.shape[0] > 0 and region.shape[1] > 0:
                patch = cv2.resize(region, (patch_size, patch_size))
                return patch
            else:
                return None
        
        # Extract center patch
        center_y = region.shape[0] // 2
        center_x = region.shape[1] // 2
        
        half_patch = patch_size // 2
        patch_y1 = max(0, center_y - half_patch)
        patch_y2 = min(region.shape[0], center_y + half_patch)
        patch_x1 = max(0, center_x - half_patch)
        patch_x2 = min(region.shape[1], center_x + half_patch)
        
        patch = region[patch_y1:patch_y2, patch_x1:patch_x2]
        
        # Ensure exactly patch_size x patch_size
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            patch = cv2.resize(patch, (patch_size, patch_size))
        
        return patch
    
    def save_scene(self, scene_desc: Dict, output_path: str):
        """
        Save scene description to JSON file.
        
        Args:
            scene_desc: Scene description
            output_path: Output JSON path
        """
        # Convert numpy types to native Python for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        scene_clean = convert_types(scene_desc)
        
        with open(output_path, 'w') as f:
            json.dump(scene_clean, f, indent=2)
        
        logger.info(f"💾 Saved scene to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Procedural Video Codec (PVC) Encoder',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input video file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output scene JSON file')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum frames to process (default: all)')
    parser.add_argument('--keyframe-interval', type=int, default=30,
                       help='Keyframe interval (default: 30)')
    parser.add_argument('--canny-low', type=int, default=50,
                       help='Canny lower threshold (default: 50)')
    parser.add_argument('--canny-high', type=int, default=150,
                       help='Canny upper threshold (default: 150)')
    
    args = parser.parse_args()
    
    # Build config
    config = {
        'keyframe_interval': args.keyframe_interval,
        'canny_low': args.canny_low,
        'canny_high': args.canny_high
    }
    
    # Encode
    encoder = PVCEncoder(config=config)
    scene_desc = encoder.encode_video(args.input, max_frames=args.max_frames)
    encoder.save_scene(scene_desc, args.output)
    
    logger.info("✅ Done!")


if __name__ == '__main__':
    main()

