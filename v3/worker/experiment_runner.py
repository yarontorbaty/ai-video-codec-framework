"""
V3.0 Experiment Runner

Executes video compression experiments by running user-provided encoding
and decoding code in a controlled environment.
"""

import os
import tempfile
import logging
import traceback
import cv2
import numpy as np
import boto3
import signal
from contextlib import contextmanager
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# S3 configuration
S3_BUCKET = 'ai-codec-v3-artifacts-580473065386'
SOURCE_VIDEO_KEY = 'reference/source.mp4'
HEVC_VIDEO_KEY = 'reference/hevc_baseline.mp4'

# Initialize S3 client
s3 = boto3.client('s3', region_name='us-east-1')

# Local cache directory for source video
CACHE_DIR = '/home/ec2-user/worker/cache'
CACHED_SOURCE_VIDEO = os.path.join(CACHE_DIR, 'source.mp4')

# Test video configuration (fallback)
TEST_VIDEO_WIDTH = 640
TEST_VIDEO_HEIGHT = 480
TEST_VIDEO_FPS = 30
TEST_VIDEO_DURATION_SEC = 2  # Short test video

# Execution timeout (2 minutes per encoding/decoding)
CODE_EXECUTION_TIMEOUT = 120


class TimeoutError(Exception):
    """Raised when code execution times out"""
    pass


@contextmanager
def timeout(seconds):
    """Context manager for timing out code execution"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution exceeded {seconds} seconds")
    
    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm and restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class ExperimentRunner:
    """Runs video compression experiments safely"""
    
    def __init__(self):
        self.test_video_cache = None
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
    
    def run_experiment(
        self,
        experiment_id: str,
        encoding_code: str,
        decoding_code: str
    ) -> Dict:
        """
        Run a complete compression experiment
        
        Returns:
            {
                'status': 'success' | 'failed',
                'original_path': str,
                'compressed_path': str,
                'reconstructed_path': str,
                'error': str (if failed)
            }
        """
        try:
            # Create unique temp paths
            temp_dir = tempfile.mkdtemp(prefix=f"exp_{experiment_id}_")
            original_path = os.path.join(temp_dir, "original.mp4")
            compressed_path = os.path.join(temp_dir, "compressed.bin")
            reconstructed_path = os.path.join(temp_dir, "reconstructed.mp4")
            
            # Create test video
            logger.info(f"📹 Creating test video...")
            self._create_test_video(original_path)
            
            # Read test video
            cap = cv2.VideoCapture(original_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            
            logger.info(f"📹 Test video: {len(frames)} frames")
            
            # Execute encoding
            logger.info(f"🔧 Executing encoding code...")
            encoding_result = self._execute_encoding(
                encoding_code,
                frames,
                compressed_path
            )
            
            if not encoding_result['success']:
                return {
                    'status': 'failed',
                    'error': f"Encoding failed: {encoding_result['error']}",
                    'original_path': original_path,
                    'compressed_path': None,
                    'reconstructed_path': None
                }
            
            # Execute decoding
            logger.info(f"🔧 Executing decoding code...")
            decoding_result = self._execute_decoding(
                decoding_code,
                compressed_path,
                reconstructed_path,
                len(frames)
            )
            
            if not decoding_result['success']:
                return {
                    'status': 'failed',
                    'error': f"Decoding failed: {decoding_result['error']}",
                    'original_path': original_path,
                    'compressed_path': compressed_path,
                    'reconstructed_path': None
                }
            
            # Verify outputs
            if not os.path.exists(reconstructed_path):
                return {
                    'status': 'failed',
                    'error': 'Reconstructed video was not created',
                    'original_path': original_path,
                    'compressed_path': compressed_path,
                    'reconstructed_path': None
                }
            
            return {
                'status': 'success',
                'original_path': original_path,
                'compressed_path': compressed_path,
                'reconstructed_path': reconstructed_path
            }
            
        except Exception as e:
            logger.error(f"❌ Experiment error: {e}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'original_path': None,
                'compressed_path': None,
                'reconstructed_path': None
            }
    
    def _create_test_video(self, output_path: str):
        """
        Use cached source video or download from S3 if not present/changed
        
        This avoids downloading 710MB on every experiment.
        """
        import shutil
        
        try:
            # Check if we have a cached version
            if os.path.exists(CACHED_SOURCE_VIDEO):
                logger.info(f"📦 Found cached source video")
                
                # Check if S3 version has changed (compare ETag/size)
                try:
                    s3_metadata = s3.head_object(Bucket=S3_BUCKET, Key=SOURCE_VIDEO_KEY)
                    s3_size = s3_metadata['ContentLength']
                    local_size = os.path.getsize(CACHED_SOURCE_VIDEO)
                    
                    if s3_size == local_size:
                        logger.info(f"✅ Using cached source video (size match: {s3_size} bytes)")
                        shutil.copy(CACHED_SOURCE_VIDEO, output_path)
                        return
                    else:
                        logger.info(f"📥 Source video changed (S3: {s3_size}, local: {local_size}), re-downloading...")
                except Exception as e:
                    logger.warning(f"⚠️ Could not check S3 metadata: {e}, using cached version")
                    shutil.copy(CACHED_SOURCE_VIDEO, output_path)
                    return
            
            # Download from S3 and cache it
            logger.info(f"📥 Downloading source video from S3 (710MB, first time only)...")
            s3.download_file(S3_BUCKET, SOURCE_VIDEO_KEY, CACHED_SOURCE_VIDEO)
            logger.info(f"✅ Source video downloaded and cached")
            
            # Copy to working directory
            shutil.copy(CACHED_SOURCE_VIDEO, output_path)
            return
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to download/cache source video: {e}")
            logger.info(f"📹 Creating fallback test video...")
        
        # Fallback: Create simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            TEST_VIDEO_FPS,
            (TEST_VIDEO_WIDTH, TEST_VIDEO_HEIGHT)
        )
        
        num_frames = TEST_VIDEO_FPS * TEST_VIDEO_DURATION_SEC
        
        for i in range(num_frames):
            # Create frame with moving gradient
            frame = np.zeros((TEST_VIDEO_HEIGHT, TEST_VIDEO_WIDTH, 3), dtype=np.uint8)
            
            # Animated gradient
            offset = int((i / num_frames) * 255)
            for y in range(TEST_VIDEO_HEIGHT):
                for x in range(TEST_VIDEO_WIDTH):
                    frame[y, x, 0] = (x + offset) % 256  # Blue
                    frame[y, x, 1] = (y + offset) % 256  # Green
                    frame[y, x, 2] = ((x + y) + offset) % 256  # Red
            
            # Add some shapes for visual interest
            cv2.circle(frame, (320, 240), 50 + i, (255, 255, 255), 2)
            cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 255), -1)
            
            out.write(frame)
        
        out.release()
        logger.info(f"✅ Fallback test video created: {num_frames} frames")
    
    def _execute_encoding(
        self,
        code: str,
        frames: list,
        output_path: str
    ) -> Dict:
        """Execute user's encoding code with timeout"""
        try:
            # Create execution environment
            env = {}
            exec(code, env)
            
            # Find the encoding function
            if 'run_encoding_agent' in env:
                encode_func = env['run_encoding_agent']
            elif 'encode' in env:
                encode_func = env['encode']
            else:
                return {
                    'success': False,
                    'error': 'No encoding function found (expected run_encoding_agent or encode)'
                }
            
            # Execute encoding with timeout
            logger.info(f"🔧 Executing encoding with {CODE_EXECUTION_TIMEOUT}s timeout...")
            with timeout(CODE_EXECUTION_TIMEOUT):
                result = encode_func(frames, output_path)
            
            # Verify output exists
            if not os.path.exists(output_path):
                return {
                    'success': False,
                    'error': 'Encoding function did not create output file'
                }
            
            return {'success': True}
            
        except TimeoutError as e:
            logger.error(f"Encoding timeout: {e}")
            return {
                'success': False,
                'error': f"Encoding timed out after {CODE_EXECUTION_TIMEOUT} seconds"
            }
        except Exception as e:
            logger.error(f"Encoding error: {e}")
            return {
                'success': False,
                'error': f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            }
    
    def _execute_decoding(
        self,
        code: str,
        input_path: str,
        output_path: str,
        expected_frames: int
    ) -> Dict:
        """Execute user's decoding code with timeout"""
        try:
            # Create execution environment
            env = {}
            exec(code, env)
            
            # Find the decoding function
            if 'run_decoding_agent' in env:
                decode_func = env['run_decoding_agent']
            elif 'decode' in env:
                decode_func = env['decode']
            else:
                return {
                    'success': False,
                    'error': 'No decoding function found (expected run_decoding_agent or decode)'
                }
            
            # Execute decoding with timeout
            logger.info(f"🔧 Executing decoding with {CODE_EXECUTION_TIMEOUT}s timeout...")
            with timeout(CODE_EXECUTION_TIMEOUT):
                result = decode_func(input_path, output_path, expected_frames)
            
            # Verify output exists
            if not os.path.exists(output_path):
                return {
                    'success': False,
                    'error': 'Decoding function did not create output file'
                }
            
            return {'success': True}
            
        except TimeoutError as e:
            logger.error(f"Decoding timeout: {e}")
            return {
                'success': False,
                'error': f"Decoding timed out after {CODE_EXECUTION_TIMEOUT} seconds"
            }
        except Exception as e:
            logger.error(f"Decoding error: {e}")
            return {
                'success': False,
                'error': f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            }

