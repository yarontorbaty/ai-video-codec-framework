"""
Bitrate Calculator - PVC Utils

Estimates bitrate for procedural codec based on scene description size.
"""

import json
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BitrateCalculator:
    """
    Calculates bitrate and compression ratio for procedural codec.
    
    Counts bits needed to store:
    - Contour points (splines/polylines)
    - Motion parameters
    - Texture parameters  
    - Residual tiles
    """
    
    @staticmethod
    def calculate_scene_size(scene_desc: Dict) -> Dict:
        """
        Calculate storage requirements for a scene description.
        
        Args:
            scene_desc: Complete scene description (ISP)
            
        Returns:
            Dictionary with size breakdown in bytes
        """
        sizes = {
            'metadata': 0,
            'contours': 0,
            'motion': 0,
            'textures': 0,
            'residuals': 0,
            'total': 0
        }
        
        # Metadata
        metadata = scene_desc.get('metadata', {})
        sizes['metadata'] = len(json.dumps(metadata).encode('utf-8'))
        
        # Objects
        objects = scene_desc.get('objects', [])
        for obj in objects:
            # Contours (store as polylines - 2 bytes per coordinate)
            contour_seq = obj.get('contour_sequence', [])
            for contour in contour_seq:
                points = contour.get('points', [])
                # Simplified: 4 bytes per point (2 bytes x, 2 bytes y)
                sizes['contours'] += len(points) * 4
            
            # Motion (store as spline parameters or keyframes)
            motion_model = obj.get('motion_model', {})
            if motion_model:
                # Estimate based on model type
                model_type = motion_model.get('type', 'keyframes')
                if model_type == 'spline':
                    params = motion_model.get('params', {})
                    # Count spline coefficients
                    coeffs_x = params.get('coeffs_x', [])
                    coeffs_y = params.get('coeffs_y', [])
                    sizes['motion'] += (len(coeffs_x) + len(coeffs_y)) * 4  # 4 bytes per float
                else:  # keyframes
                    params = motion_model.get('params', {})
                    translations = params.get('translations', [])
                    sizes['motion'] += len(translations) * 2 * 4  # 2 floats per translation
            
            # Texture (compact parameters)
            texture = obj.get('texture', {})
            if texture:
                # Texture type (1 byte) + parameters
                sizes['textures'] += 1
                params = texture.get('params', {})
                # Count parameters (assume float32)
                sizes['textures'] += len(params) * 4
        
        # Residuals (if any)
        residuals = scene_desc.get('residuals', [])
        for residual in residuals:
            # Residual data (actual compressed tiles)
            data = residual.get('data', None)
            if data is not None:
                if isinstance(data, np.ndarray):
                    sizes['residuals'] += data.nbytes
                elif isinstance(data, bytes):
                    sizes['residuals'] += len(data)
        
        sizes['total'] = sum(sizes.values())
        
        return sizes
    
    @staticmethod
    def calculate_bitrate(scene_desc: Dict,
                         duration: float) -> Dict:
        """
        Calculate bitrate in Mbps.
        
        Args:
            scene_desc: Scene description
            duration: Video duration in seconds
            
        Returns:
            Dictionary with bitrate metrics
        """
        sizes = BitrateCalculator.calculate_scene_size(scene_desc)
        
        total_bytes = sizes['total']
        total_bits = total_bytes * 8
        
        if duration <= 0:
            duration = 1.0
        
        bitrate_bps = total_bits / duration
        bitrate_kbps = bitrate_bps / 1000
        bitrate_mbps = bitrate_kbps / 1000
        
        return {
            'total_bytes': total_bytes,
            'total_bits': total_bits,
            'duration_sec': duration,
            'bitrate_bps': bitrate_bps,
            'bitrate_kbps': bitrate_kbps,
            'bitrate_mbps': bitrate_mbps,
            'breakdown': sizes
        }
    
    @staticmethod
    def compare_to_baseline(pvc_bitrate: float,
                           baseline_bitrate: float) -> Dict:
        """
        Compare PVC bitrate to baseline (AV1).
        
        Args:
            pvc_bitrate: PVC bitrate in Mbps
            baseline_bitrate: Baseline bitrate in Mbps
            
        Returns:
            Comparison metrics
        """
        if baseline_bitrate <= 0:
            return {
                'reduction_percent': 0.0,
                'compression_ratio': 1.0,
                'pvc_mbps': pvc_bitrate,
                'baseline_mbps': baseline_bitrate
            }
        
        reduction = baseline_bitrate - pvc_bitrate
        reduction_percent = (reduction / baseline_bitrate) * 100
        compression_ratio = baseline_bitrate / (pvc_bitrate + 1e-9)
        
        return {
            'reduction_percent': reduction_percent,
            'compression_ratio': compression_ratio,
            'pvc_mbps': pvc_bitrate,
            'baseline_mbps': baseline_bitrate,
            'achieved_target': reduction_percent >= 90.0
        }
    
    @staticmethod
    def estimate_from_video(video_path: str,
                           num_objects: int,
                           avg_contour_points: int,
                           use_residuals: bool = False) -> Dict:
        """
        Estimate PVC bitrate for a video without encoding.
        
        Args:
            video_path: Input video path
            num_objects: Estimated number of objects per frame
            avg_contour_points: Average points per contour
            use_residuals: Whether residuals are needed
            
        Returns:
            Estimated bitrate metrics
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        duration = frame_count / fps if fps > 0 else 1.0
        
        # Estimate size per frame
        # Contours: num_objects * avg_contour_points * 4 bytes
        contour_bytes_per_frame = num_objects * avg_contour_points * 4
        
        # Motion: num_objects * 8 bytes per frame (dx, dy as floats)
        motion_bytes_per_frame = num_objects * 8
        
        # Textures: num_objects * 20 bytes (type + ~4 params)
        texture_bytes_per_frame = num_objects * 20
        
        # Residuals: estimate 10% of frame size if used
        residual_bytes_per_frame = 0
        if use_residuals:
            # Assume 10% of pixels need residuals
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            pixels_per_frame = width * height
            residual_bytes_per_frame = int(pixels_per_frame * 0.1 * 3)  # 3 bytes per pixel
        
        total_bytes_per_frame = (
            contour_bytes_per_frame +
            motion_bytes_per_frame +
            texture_bytes_per_frame +
            residual_bytes_per_frame
        )
        
        # Total size (with keyframe optimization - store full contours every 30 frames)
        keyframe_interval = 30
        num_keyframes = frame_count // keyframe_interval
        num_interframes = frame_count - num_keyframes
        
        total_bytes = (
            num_keyframes * total_bytes_per_frame +
            num_interframes * (motion_bytes_per_frame + texture_bytes_per_frame + residual_bytes_per_frame)
        )
        
        bitrate_mbps = (total_bytes * 8) / (duration * 1_000_000)
        
        return {
            'estimated_bytes': total_bytes,
            'estimated_bitrate_mbps': bitrate_mbps,
            'duration_sec': duration,
            'frame_count': frame_count,
            'fps': fps,
            'assumptions': {
                'num_objects': num_objects,
                'avg_contour_points': avg_contour_points,
                'use_residuals': use_residuals,
                'keyframe_interval': keyframe_interval
            }
        }


# Example usage
if __name__ == "__main__":
    # Test scene size calculation
    test_scene = {
        'metadata': {'fps': 30, 'duration': 1.0},
        'objects': [
            {
                'contour_sequence': [
                    {'points': [[i, i] for i in range(20)]}
                    for _ in range(30)
                ],
                'motion_model': {
                    'type': 'keyframes',
                    'params': {
                        'translations': [[1.0, 1.0] for _ in range(30)]
                    }
                },
                'texture': {
                    'type': 'perlin',
                    'params': {'scale': 10, 'seed': 42}
                }
            }
        ],
        'residuals': []
    }
    
    sizes = BitrateCalculator.calculate_scene_size(test_scene)
    print("Scene size breakdown:")
    for key, value in sizes.items():
        print(f"  {key}: {value} bytes")
    
    bitrate = BitrateCalculator.calculate_bitrate(test_scene, duration=1.0)
    print(f"\nBitrate: {bitrate['bitrate_mbps']:.3f} Mbps")
    
    # Compare to baseline
    comparison = BitrateCalculator.compare_to_baseline(
        pvc_bitrate=0.5,
        baseline_bitrate=5.0
    )
    print(f"\nCompression: {comparison['reduction_percent']:.1f}% reduction")
    print(f"Target achieved: {comparison['achieved_target']}")

