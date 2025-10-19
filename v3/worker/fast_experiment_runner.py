"""
V3.0 Fast Experiment Runner - Optimized for 10,000+ experiments/hour

Changes from standard runner:
1. Tiny test videos (10 frames @ 64x64 pixels) instead of HD
2. Batch processing (100 experiments at once)
3. Simple MSE instead of PSNR/SSIM for initial filtering
4. In-memory processing (no disk I/O)
5. Minimal logging overhead
"""

import logging
import traceback
import cv2
import numpy as np
import time
from typing import Dict, List, Tuple
import signal
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Fast test video configuration
FAST_VIDEO_WIDTH = 64
FAST_VIDEO_HEIGHT = 64
FAST_VIDEO_FRAMES = 10

# Execution timeout (10 seconds per encoding/decoding for tiny videos)
CODE_EXECUTION_TIMEOUT = 10


class TimeoutError(Exception):
    """Raised when code execution times out"""
    pass


@contextmanager
def timeout(seconds):
    """Context manager for timing out code execution"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution exceeded {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class FastExperimentRunner:
    """Runs video compression experiments at high speed"""
    
    def __init__(self):
        # Pre-generate test video frames once
        self.test_frames = self._generate_test_frames()
        logger.info(f"🚀 Fast runner initialized: {FAST_VIDEO_FRAMES} frames @ {FAST_VIDEO_WIDTH}x{FAST_VIDEO_HEIGHT}")
    
    def run_batch(
        self,
        experiments: List[Dict]
    ) -> List[Dict]:
        """
        Run a batch of experiments quickly
        
        Args:
            experiments: List of {
                'experiment_id': str,
                'encoding_code': str,
                'decoding_code': str
            }
        
        Returns:
            List of {
                'experiment_id': str,
                'status': 'success' | 'failed',
                'mse': float,
                'compression_ratio': float,
                'time_ms': int,
                'error': str (if failed)
            }
        """
        results = []
        start_time = time.time()
        
        for i, exp in enumerate(experiments):
            result = self._run_single_experiment(exp)
            results.append(result)
            
            # Log progress every 10 experiments
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                logger.info(f"⚡ Progress: {i+1}/{len(experiments)} ({rate:.1f} exp/sec)")
        
        total_time = time.time() - start_time
        rate = len(experiments) / total_time
        logger.info(f"✅ Batch complete: {len(experiments)} experiments in {total_time:.1f}s ({rate:.1f} exp/sec)")
        
        return results
    
    def _run_single_experiment(self, exp: Dict) -> Dict:
        """Run a single experiment (optimized)"""
        experiment_id = exp['experiment_id']
        start_time = time.time()
        
        try:
            # Execute encoding (in-memory)
            encoding_result = self._execute_encoding_fast(
                exp['encoding_code'],
                self.test_frames
            )
            
            if not encoding_result['success']:
                return {
                    'experiment_id': experiment_id,
                    'status': 'failed',
                    'error': f"Encoding: {encoding_result['error']}",
                    'mse': None,
                    'compression_ratio': None,
                    'time_ms': int((time.time() - start_time) * 1000)
                }
            
            compressed_data = encoding_result['data']
            
            # Execute decoding (in-memory)
            decoding_result = self._execute_decoding_fast(
                exp['decoding_code'],
                compressed_data
            )
            
            if not decoding_result['success']:
                return {
                    'experiment_id': experiment_id,
                    'status': 'failed',
                    'error': f"Decoding: {decoding_result['error']}",
                    'mse': None,
                    'compression_ratio': None,
                    'time_ms': int((time.time() - start_time) * 1000)
                }
            
            reconstructed_frames = decoding_result['frames']
            
            # Calculate MSE (fast metric)
            mse = self._calculate_mse(self.test_frames, reconstructed_frames)
            
            # Calculate compression ratio
            original_size = len(self.test_frames) * FAST_VIDEO_WIDTH * FAST_VIDEO_HEIGHT * 3
            compressed_size = len(compressed_data)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            return {
                'experiment_id': experiment_id,
                'status': 'success',
                'mse': float(mse),
                'compression_ratio': float(compression_ratio),
                'time_ms': elapsed_ms,
                'compressed_size': compressed_size,
                'error': None
            }
            
        except Exception as e:
            return {
                'experiment_id': experiment_id,
                'status': 'failed',
                'error': str(e),
                'mse': None,
                'compression_ratio': None,
                'time_ms': int((time.time() - start_time) * 1000)
            }
    
    def _generate_test_frames(self) -> List[np.ndarray]:
        """Generate test video frames once (reused for all experiments)"""
        frames = []
        
        for i in range(FAST_VIDEO_FRAMES):
            # Create frame with interesting content
            frame = np.zeros((FAST_VIDEO_HEIGHT, FAST_VIDEO_WIDTH, 3), dtype=np.uint8)
            
            # Animated gradient
            offset = int((i / FAST_VIDEO_FRAMES) * 255)
            for y in range(FAST_VIDEO_HEIGHT):
                for x in range(FAST_VIDEO_WIDTH):
                    frame[y, x, 0] = (x * 4 + offset) % 256  # Blue
                    frame[y, x, 1] = (y * 4 + offset) % 256  # Green
                    frame[y, x, 2] = ((x + y) * 2 + offset) % 256  # Red
            
            # Add some shapes
            cv2.circle(frame, (32, 32), 10 + i, (255, 255, 255), 1)
            cv2.rectangle(frame, (10, 10), (25, 25), (0, 255, 255), -1)
            
            frames.append(frame)
        
        return frames
    
    def _execute_encoding_fast(self, code: str, frames: List[np.ndarray]) -> Dict:
        """Execute encoding code quickly (in-memory)"""
        try:
            # Create execution environment
            env = {}
            exec(code, env)
            
            # Find the encoding function
            if 'encode' in env:
                encode_func = env['encode']
            elif 'run_encoding_agent' in env:
                encode_func = env['run_encoding_agent']
            else:
                return {
                    'success': False,
                    'error': 'No encoding function found'
                }
            
            # Execute with timeout
            with timeout(CODE_EXECUTION_TIMEOUT):
                # Expect function to return bytes directly
                result = encode_func(frames)
            
            if not isinstance(result, (bytes, bytearray)):
                return {
                    'success': False,
                    'error': f'Encoding must return bytes, got {type(result)}'
                }
            
            return {
                'success': True,
                'data': bytes(result)
            }
            
        except TimeoutError:
            return {
                'success': False,
                'error': f'Encoding timeout after {CODE_EXECUTION_TIMEOUT}s'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'{type(e).__name__}: {str(e)}'
            }
    
    def _execute_decoding_fast(self, code: str, compressed_data: bytes) -> Dict:
        """Execute decoding code quickly (in-memory)"""
        try:
            # Create execution environment
            env = {}
            exec(code, env)
            
            # Find the decoding function
            if 'decode' in env:
                decode_func = env['decode']
            elif 'run_decoding_agent' in env:
                decode_func = env['run_decoding_agent']
            else:
                return {
                    'success': False,
                    'error': 'No decoding function found'
                }
            
            # Execute with timeout
            with timeout(CODE_EXECUTION_TIMEOUT):
                # Expect function to return list of frames
                result = decode_func(compressed_data, FAST_VIDEO_FRAMES)
            
            if not isinstance(result, list):
                return {
                    'success': False,
                    'error': f'Decoding must return list of frames, got {type(result)}'
                }
            
            if len(result) != FAST_VIDEO_FRAMES:
                return {
                    'success': False,
                    'error': f'Expected {FAST_VIDEO_FRAMES} frames, got {len(result)}'
                }
            
            # Validate frame dimensions
            for i, frame in enumerate(result):
                if not isinstance(frame, np.ndarray):
                    return {
                        'success': False,
                        'error': f'Frame {i} is not numpy array'
                    }
                if frame.shape != (FAST_VIDEO_HEIGHT, FAST_VIDEO_WIDTH, 3):
                    return {
                        'success': False,
                        'error': f'Frame {i} has wrong shape: {frame.shape}'
                    }
            
            return {
                'success': True,
                'frames': result
            }
            
        except TimeoutError:
            return {
                'success': False,
                'error': f'Decoding timeout after {CODE_EXECUTION_TIMEOUT}s'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'{type(e).__name__}: {str(e)}'
            }
    
    def _calculate_mse(self, original_frames: List[np.ndarray], reconstructed_frames: List[np.ndarray]) -> float:
        """Calculate Mean Squared Error (fast quality metric)"""
        if len(original_frames) != len(reconstructed_frames):
            return float('inf')
        
        total_mse = 0.0
        for orig, recon in zip(original_frames, reconstructed_frames):
            mse = np.mean((orig.astype(float) - recon.astype(float)) ** 2)
            total_mse += mse
        
        return total_mse / len(original_frames)

