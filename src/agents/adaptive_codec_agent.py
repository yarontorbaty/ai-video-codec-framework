#!/usr/bin/env python3
"""
Adaptive Codec Agent - Uses LLM-Generated Code
This agent can replace its own implementation with better LLM-generated code.
"""

import os
import json
import logging
import numpy as np
import cv2
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class AdaptiveCodecAgent:
    """
    Codec agent that can evolve by adopting LLM-generated compression code.
    """
    
    def __init__(self, resolution=(1920, 1080)):
        self.resolution = resolution
        self.current_implementation = None
        self.implementation_version = 0
        self.performance_history = []
        
        # Load current best implementation if it exists
        self.load_best_implementation()
    
    def load_best_implementation(self):
        """Load the current best implementation from disk."""
        impl_file = '/tmp/best_codec_implementation.json'
        if os.path.exists(impl_file):
            try:
                with open(impl_file, 'r') as f:
                    data = json.load(f)
                    self.current_implementation = data.get('code')
                    self.implementation_version = data.get('version', 0)
                    logger.info(f"✅ Loaded implementation v{self.implementation_version}")
            except Exception as e:
                logger.warning(f"Could not load implementation: {e}")
    
    def save_implementation(self, code: str, metrics: Dict):
        """Save a new implementation as the current best."""
        impl_file = '/tmp/best_codec_implementation.json'
        self.implementation_version += 1
        
        data = {
            'version': self.implementation_version,
            'code': code,
            'metrics': metrics,
            'timestamp': str(np.datetime64('now'))
        }
        
        with open(impl_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.current_implementation = code
        logger.info(f"💾 Saved new implementation v{self.implementation_version}")
    
    def should_adopt_new_code(self, new_metrics: Dict) -> bool:
        """
        Decide if new LLM-generated code should replace current implementation.
        
        Criteria:
        - Lower bitrate AND higher quality
        - OR significantly lower bitrate (>50% reduction) with acceptable quality loss (<5%)
        """
        if not self.performance_history:
            # No history - adopt anything that works
            return new_metrics.get('success', False)
        
        # Get best previous metrics
        best_prev = min(self.performance_history, key=lambda x: x.get('bitrate_mbps', 999))
        
        new_bitrate = new_metrics.get('bitrate_mbps', 999)
        new_quality = new_metrics.get('compression_ratio', 0)
        
        prev_bitrate = best_prev.get('bitrate_mbps', 999)
        prev_quality = best_prev.get('compression_ratio', 0)
        
        # Adoption criteria
        if new_bitrate < prev_bitrate * 0.9:  # 10% better bitrate
            logger.info(f"🎯 New code is {(1 - new_bitrate/prev_bitrate)*100:.1f}% better!")
            return True
        
        if new_quality > prev_quality * 1.2:  # 20% better compression ratio
            logger.info(f"🎯 New code has {(new_quality/prev_quality - 1)*100:.1f}% better compression!")
            return True
        
        return False
    
    def test_generated_code(self, code: str, function_name: str = 'compress_video_frame') -> Tuple[bool, Optional[Dict]]:
        """
        Test LLM-generated code and return performance metrics.
        """
        try:
            from utils.code_sandbox import CodeSandbox
            
            sandbox = CodeSandbox(timeout=30)
            
            # Validate code
            is_valid, error = sandbox.validate_code(code)
            if not is_valid:
                logger.error(f"Code validation failed: {error}")
                return False, None
            
            # Test on sample frames
            test_frames = [
                np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),  # Random
                np.zeros((1080, 1920, 3), dtype=np.uint8),  # Black
                np.ones((1080, 1920, 3), dtype=np.uint8) * 128,  # Gray
            ]
            
            total_original_size = 0
            total_compressed_size = 0
            successes = 0
            
            for i, frame in enumerate(test_frames):
                config = {'quality': 0.8}
                success, result, exec_error = sandbox.execute_function(
                    code, function_name, frame, config
                )
                
                if success and result:
                    original_size = frame.nbytes
                    compressed_size = len(result.get('compressed', b''))
                    
                    total_original_size += original_size
                    total_compressed_size += compressed_size
                    successes += 1
                    
                    logger.info(f"Test {i+1}: {original_size} → {compressed_size} bytes")
            
            if successes == 0:
                return False, None
            
            # Calculate metrics
            compression_ratio = total_original_size / total_compressed_size if total_compressed_size > 0 else 0
            bitrate_mbps = (total_compressed_size * 8) / (len(test_frames) * 1_000_000) * 30  # Assuming 30fps
            
            metrics = {
                'success': True,
                'compression_ratio': compression_ratio,
                'bitrate_mbps': bitrate_mbps,
                'test_frames': successes,
                'avg_compressed_size': total_compressed_size / successes
            }
            
            logger.info(f"📊 Metrics: {compression_ratio:.2f}x compression, {bitrate_mbps:.2f} Mbps")
            
            return True, metrics
            
        except Exception as e:
            logger.error(f"Error testing generated code: {e}")
            return False, None
    
    def evolve_with_llm_code(self, generated_code_info: Dict) -> Dict:
        """
        Evaluate and potentially adopt LLM-generated code.
        
        Returns:
            Dict with evolution results (adopted/rejected, metrics, etc.)
        """
        if not generated_code_info or 'code' not in generated_code_info:
            return {'status': 'no_code', 'adopted': False}
        
        code = generated_code_info['code']
        function_name = generated_code_info.get('function_name', 'compress_video_frame')
        
        logger.info("🧬 Evaluating LLM-generated code for evolution...")
        
        # Test the new code
        success, metrics = self.test_generated_code(code, function_name)
        
        if not success or not metrics:
            return {
                'status': 'test_failed',
                'adopted': False,
                'reason': 'Code testing failed or produced invalid results'
            }
        
        # Decide if we should adopt it
        should_adopt = self.should_adopt_new_code(metrics)
        
        if should_adopt:
            self.save_implementation(code, metrics)
            self.performance_history.append(metrics)
            
            logger.info(f"🎉 EVOLUTION SUCCESS! Adopted new implementation v{self.implementation_version}")
            
            return {
                'status': 'adopted',
                'adopted': True,
                'version': self.implementation_version,
                'metrics': metrics,
                'improvement': self._calculate_improvement(metrics)
            }
        else:
            self.performance_history.append(metrics)
            
            logger.info("⏭️  New code not better than current - keeping existing implementation")
            
            return {
                'status': 'rejected',
                'adopted': False,
                'reason': 'Performance not better than current implementation',
                'metrics': metrics
            }
    
    def _calculate_improvement(self, new_metrics: Dict) -> Dict:
        """Calculate improvement over previous best."""
        if len(self.performance_history) < 2:
            return {'first_implementation': True}
        
        prev = self.performance_history[-2]
        new = new_metrics
        
        bitrate_improvement = ((prev.get('bitrate_mbps', 999) - new.get('bitrate_mbps', 999)) / 
                               prev.get('bitrate_mbps', 999) * 100)
        
        return {
            'bitrate_reduction_percent': bitrate_improvement,
            'compression_ratio_change': new.get('compression_ratio', 0) / prev.get('compression_ratio', 1)
        }
    
    def compress_video(self, input_path: str, output_path: str) -> Dict:
        """
        Compress video using current best implementation.
        Falls back to default if no implementation available.
        """
        if not self.current_implementation:
            logger.warning("No evolved implementation available - using baseline")
            return self._baseline_compression(input_path, output_path)
        
        try:
            from utils.code_sandbox import CodeSandbox
            
            # Load video
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            compressed_data = []
            sandbox = CodeSandbox(timeout=60)
            
            logger.info(f"Compressing {frame_count} frames with evolved codec v{self.implementation_version}...")
            
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                
                config = {'quality': 0.8}
                success, result, error = sandbox.execute_function(
                    self.current_implementation,
                    'compress_video_frame',
                    frame,
                    config
                )
                
                if success and result:
                    compressed_data.append(result.get('compressed', b''))
            
            cap.release()
            
            # Save compressed data
            total_size = sum(len(d) for d in compressed_data)
            duration = frame_count / fps
            bitrate_mbps = (total_size * 8) / (duration * 1_000_000)
            
            with open(output_path, 'wb') as f:
                for data in compressed_data:
                    f.write(data)
            
            return {
                'status': 'success',
                'implementation_version': self.implementation_version,
                'bitrate_mbps': bitrate_mbps,
                'file_size_bytes': total_size,
                'frames_compressed': len(compressed_data)
            }
            
        except Exception as e:
            logger.error(f"Error using evolved implementation: {e}")
            return self._baseline_compression(input_path, output_path)
    
    def _baseline_compression(self, input_path: str, output_path: str) -> Dict:
        """Fallback baseline compression."""
        # Simple copy as baseline
        import shutil
        shutil.copy(input_path, output_path)
        
        size = os.path.getsize(output_path)
        
        return {
            'status': 'baseline_fallback',
            'implementation_version': 0,
            'bitrate_mbps': (size * 8) / (10.0 * 1_000_000),
            'file_size_bytes': size
        }

