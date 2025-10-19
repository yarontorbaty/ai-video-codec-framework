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
        
        # Analyze textures (simplified - assign procedural textures based on region properties)
        logger.info("🎨 Analyzing textures...")
        self._assign_textures(tracked_objects)
        
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
    
    def _assign_textures(self, objects: List[Dict]):
        """
        Assign procedural textures to objects based on their properties.
        
        For prototype, use simple heuristics:
        - Small objects: solid colors
        - Large objects: procedural textures
        """
        for obj in objects:
            if not obj.get('contour_sequence'):
                continue
            
            # Get average area
            areas = [c.get('area', 0) for c in obj['contour_sequence']]
            avg_area = np.mean(areas) if areas else 0
            
            # Simple texture assignment
            if avg_area < 1000:
                # Small object - solid color
                obj['texture'] = {
                    'type': 'solid',
                    'color': [
                        np.random.rand(),
                        np.random.rand(),
                        np.random.rand()
                    ],
                    'params': {}
                }
            elif avg_area < 10000:
                # Medium object - Perlin noise
                obj['texture'] = {
                    'type': 'perlin',
                    'params': {
                        'scale': np.random.uniform(5, 20),
                        'octaves': 4,
                        'persistence': 0.5,
                        'lacunarity': 2.0,
                        'seed': np.random.randint(0, 10000)
                    },
                    'color_map': [
                        np.random.rand(),
                        np.random.rand(),
                        np.random.rand()
                    ]
                }
            else:
                # Large object - fBM or Worley
                texture_type = np.random.choice(['fbm', 'worley'])
                
                if texture_type == 'fbm':
                    obj['texture'] = {
                        'type': 'fbm',
                        'params': {
                            'base_scale': np.random.uniform(10, 30),
                            'octaves': 6,
                            'persistence': 0.5,
                            'lacunarity': 2.0,
                            'seed': np.random.randint(0, 10000)
                        },
                        'color_map': [
                            np.random.rand(),
                            np.random.rand(),
                            np.random.rand()
                        ]
                    }
                else:  # worley
                    obj['texture'] = {
                        'type': 'worley',
                        'params': {
                            'num_points': np.random.randint(10, 30),
                            'distance_func': 'euclidean',
                            'seed': np.random.randint(0, 10000)
                        },
                        'color_map': [
                            np.random.rand(),
                            np.random.rand(),
                            np.random.rand()
                        ]
                    }
    
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

