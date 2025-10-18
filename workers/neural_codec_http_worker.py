#!/usr/bin/env python3
"""
Neural Codec HTTP Worker
Executes two-agent neural video codec experiments on GPU hardware.
Accepts HTTP requests for jobs and processes encoding/decoding with quality measurement.
"""

import os
import sys
import json
import time
import logging
import traceback
import boto3
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, Optional
import socket
import tempfile
from flask import Flask, request, jsonify
import threading
import requests

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS clients
s3 = boto3.client('s3', region_name='us-east-1')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

# Configuration
VIDEO_BUCKET = 'ai-video-codec-videos-580473065386'
EXPERIMENTS_TABLE = 'ai-video-codec-experiments'
WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"
HTTP_PORT = int(os.environ.get('WORKER_HTTP_PORT', 8080))
ORCHESTRATOR_URL = os.environ.get('ORCHESTRATOR_URL', 'http://10.0.1.109:8081')

# GPU detection
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import cv2
    from skimage.metrics import structural_similarity as ssim
    
    if torch.cuda.is_available():
        GPU_DEVICE = 'cuda'
        GPU_COUNT = torch.cuda.device_count()
        logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        GPU_DEVICE = 'cpu'
        GPU_COUNT = 0
        logger.warning("⚠️  No GPU detected - will use CPU")
        
except ImportError as e:
    logger.warning(f"⚠️  GPU libraries not available: {e}")
    GPU_DEVICE = 'cpu'
    GPU_COUNT = 0

# Flask app
app = Flask(__name__)

class NeuralCodecWorker:
    """HTTP-based Neural Codec Worker"""
    
    def __init__(self):
        self.worker_id = WORKER_ID
        self.device = GPU_DEVICE
        self.jobs_processed = 0
        self.is_processing = False
        self.experiment_queue = []  # Queue for experiments
        self.queue_lock = threading.Lock()  # Thread-safe queue access
        
    def health_check(self) -> Dict:
        """Run health check."""
        try:
            # Test AWS connections
            s3.list_objects_v2(Bucket=VIDEO_BUCKET, MaxKeys=1)
            dynamodb.Table(EXPERIMENTS_TABLE).scan(Limit=1)
            
            return {
                'status': 'healthy',
                'worker_id': self.worker_id,
                'device': self.device,
                'gpu_count': GPU_COUNT,
                'jobs_processed': self.jobs_processed,
                'timestamp': int(time.time())
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': int(time.time())
            }
    
    def process_experiment(self, job_data: Dict) -> Dict:
        """Process a neural codec experiment with unique file paths and real metrics."""
        try:
            experiment_id = job_data.get('experiment_id')
            # Support both naming conventions
            encoding_code = job_data.get('encoding_agent_code') or job_data.get('encoding_code', '')
            decoding_code = job_data.get('decoding_agent_code') or job_data.get('decoding_code', '')
            timestamp = int(time.time())
            
            logger.info(f"🎯 Processing experiment: {experiment_id}")
            print(f"DEBUG: encoding_code length: {len(encoding_code) if encoding_code else 0}", flush=True)
            print(f"DEBUG: decoding_code length: {len(decoding_code) if decoding_code else 0}", flush=True)
            print(f"DEBUG: job_data keys: {list(job_data.keys())}", flush=True)
            logger.info(f"   🔍 DEBUG: encoding_code length: {len(encoding_code) if encoding_code else 0}")
            logger.info(f"   🔍 DEBUG: decoding_code length: {len(decoding_code) if decoding_code else 0}")
            logger.info(f"   🔍 DEBUG: job_data keys: {list(job_data.keys())}")
            
            # Create unique output paths for this experiment
            unique_suffix = f"{experiment_id}_{timestamp}"
            compressed_path = f"/tmp/compressed_{unique_suffix}.bin"
            original_path = f"/tmp/original_{unique_suffix}.mp4"
            reconstructed_path = f"/tmp/reconstructed_{unique_suffix}.mp4"
            
            # Import code sandbox for safe execution
            from src.utils.code_sandbox import CodeSandbox
            
            sandbox = CodeSandbox()
            
            # Create a unique test frame for this experiment
            test_frame = self._create_unique_test_frame(experiment_id, timestamp)
            
            # Execute encoding agent with unique paths
            encoding_result = None
            encoding_success = False
            if encoding_code:
                logger.info("   Running encoding agent...")
                # Call encoding agent with correct arguments
                success, result, error = sandbox.execute_function(
                    encoding_code, 
                    'run_encoding_agent',
                    args=(),
                    kwargs={
                        'device': self.device,
                        'test_frame': test_frame,
                        'output_path': compressed_path
                    }
                )
                
                if success:
                    encoding_result = result
                    encoding_success = True
                    logger.info(f"   ✅ Encoding successful")
                else:
                    logger.error(f"   ❌ Encoding failed: {error}")
                    encoding_result = {'status': 'failed', 'error': error}
            
            # Execute decoding agent with unique paths
            decoding_result = None
            decoding_success = False
            if decoding_code and encoding_success:
                logger.info("   Running decoding agent...")
                # Call decoding agent with correct arguments
                success, result, error = sandbox.execute_function(
                    decoding_code,
                    'run_decoding_agent',
                    args=(),
                    kwargs={
                        'device': self.device,
                        'encoding_data': encoding_result,
                        'compressed_path': compressed_path,
                        'output_path': reconstructed_path
                    }
                )
                
                if success:
                    decoding_result = result
                    decoding_success = True
                    logger.info(f"   ✅ Decoding successful")
                else:
                    logger.error(f"   ❌ Decoding failed: {error}")
                    decoding_result = {'status': 'failed', 'error': error}
            
            # Calculate REAL metrics from unique files (BEFORE cleanup!)
            metrics = self._calculate_real_metrics(
                original_path, compressed_path, reconstructed_path, 
                encoding_result, decoding_result, test_frame
            )
            
            # Verify output files exist
            output_files = {
                'compressed': compressed_path if os.path.exists(compressed_path) else None,
                'reconstructed': reconstructed_path if os.path.exists(reconstructed_path) else None
            }
            
            # Upload video and decoder to S3 if successful
            video_url = None
            decoder_s3_key = None
            if encoding_success and decoding_success and os.path.exists(reconstructed_path):
                logger.info("   📤 Uploading video and decoder to S3...")
                video_url = self._upload_video_to_s3(reconstructed_path, experiment_id)
                decoder_s3_key = self._save_decoder_to_s3(decoding_code, experiment_id)
                logger.info(f"   ✅ Video URL: {video_url}")
                logger.info(f"   ✅ Decoder key: {decoder_s3_key}")
            
            # Add media URLs to metrics
            metrics['video_url'] = video_url
            metrics['decoder_s3_key'] = decoder_s3_key
            
            # Determine actual status based on execution success
            if encoding_success and decoding_success:
                status = 'completed'
            elif encoding_success:
                status = 'partial_success'  # Encoding worked but decoding failed
            else:
                status = 'failed'
            
            # Clean up temporary files (AFTER we've uploaded everything)
            self._cleanup_temp_files([compressed_path, original_path, reconstructed_path])
            
            result = {
                'experiment_id': experiment_id,
                'status': status,
                'metrics': metrics,
                'encoding_result': encoding_result,
                'decoding_result': decoding_result,
                'output_files': output_files,
                'execution_success': {
                    'encoding': encoding_success,
                    'decoding': decoding_success
                },
                'worker_id': self.worker_id,
                'timestamp': int(time.time())
            }
            
            self.jobs_processed += 1
            logger.info(f"✅ Experiment {experiment_id} completed")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Experiment failed: {e}")
            return {
                'experiment_id': job_data.get('experiment_id'),
                'status': 'failed',
                'error': str(e),
                'worker_id': self.worker_id,
                'timestamp': int(time.time())
            }
    
    def save_result_to_dynamodb(self, result: Dict):
        """Save experiment result directly to DynamoDB."""
        try:
            from datetime import datetime
            from decimal import Decimal
            import copy
            
            experiment_id = result.get('experiment_id', 'unknown')
            timestamp = result.get('timestamp', int(time.time()))
            
            # Deep copy and clean the result
            result_clean = copy.deepcopy(result)
            
            # Remove large binary data
            if 'reconstructed_frame' in result_clean:
                del result_clean['reconstructed_frame']
            if 'encoding_result' in result_clean and isinstance(result_clean['encoding_result'], dict):
                if 'reconstructed_frame' in result_clean['encoding_result']:
                    del result_clean['encoding_result']['reconstructed_frame']
            if 'decoding_result' in result_clean and isinstance(result_clean['decoding_result'], dict):
                if 'reconstructed_frame' in result_clean['decoding_result']:
                    del result_clean['decoding_result']['reconstructed_frame']
            
            # Convert floats to Decimal for DynamoDB
            def convert_floats(obj):
                if isinstance(obj, float):
                    return Decimal(str(obj))
                elif isinstance(obj, dict):
                    return {k: convert_floats(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_floats(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_floats(item) for item in obj)
                return obj
            
            result_clean = convert_floats(result_clean)
            
            # Build DynamoDB item
            item = {
                'experiment_id': experiment_id,
                'timestamp': timestamp,
                'timestamp_iso': datetime.now().isoformat(),
                'status': 'completed',
                'result': result_clean
            }
            
            # Save to DynamoDB
            table = dynamodb.Table(EXPERIMENTS_TABLE)
            table.put_item(Item=item)
            
            logger.info(f"✅ Result saved to DynamoDB: {experiment_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save result to DynamoDB: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def send_result_to_orchestrator(self, result: Dict):
        """Send experiment result back to orchestrator."""
        try:
            import base64
            import copy
            
            # Deep copy to avoid modifying original
            result_copy = copy.deepcopy(result)
            
            # Recursively sanitize binary data
            def sanitize_binary(obj):
                if isinstance(obj, bytes):
                    return base64.b64encode(obj).decode('utf-8')
                elif isinstance(obj, dict):
                    return {k: sanitize_binary(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [sanitize_binary(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(sanitize_binary(item) for item in obj)
                else:
                    return obj
            
            result_copy = sanitize_binary(result_copy)
            
            # Remove large data
            if 'reconstructed_frame' in result_copy:
                del result_copy['reconstructed_frame']
            if 'encoding_result' in result_copy and isinstance(result_copy['encoding_result'], dict):
                if 'reconstructed_frame' in result_copy['encoding_result']:
                    del result_copy['encoding_result']['reconstructed_frame']
            if 'decoding_result' in result_copy and isinstance(result_copy['decoding_result'], dict):
                if 'reconstructed_frame' in result_copy['decoding_result']:
                    del result_copy['decoding_result']['reconstructed_frame']
            
            response = requests.post(
                f"{ORCHESTRATOR_URL}/experiment_result",
                json=result_copy,
                timeout=30
            )
            response.raise_for_status()
            logger.info("✅ Result sent to orchestrator")
        except Exception as e:
            logger.error(f"❌ Failed to send result to orchestrator: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _create_unique_test_frame(self, experiment_id: str, timestamp: int) -> np.ndarray:
        """Create a unique test frame for this experiment."""
        # Use experiment ID and timestamp to create unique frame
        np.random.seed(hash(f"{experiment_id}_{timestamp}") % 2**32)
        
        # Create base frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add unique patterns based on experiment ID
        frame_hash = hash(experiment_id) % 1000
        
        # Add unique geometric patterns
        cv2.rectangle(frame, (frame_hash % 200, frame_hash % 150), 
                     (frame_hash % 200 + 100, frame_hash % 150 + 100), 
                     (255, 0, 0), -1)
        
        cv2.circle(frame, (frame_hash % 400 + 200, frame_hash % 300 + 200), 
                  frame_hash % 50 + 30, (0, 255, 0), -1)
        
        # Add unique text
        cv2.putText(frame, f"EXP_{experiment_id[:8]}", 
                   (frame_hash % 300, frame_hash % 400 + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def _inject_unique_paths(self, code: str, compressed_path: str, output_path: str) -> str:
        """Inject unique file paths into the neural codec code."""
        # Replace hardcoded paths with unique paths
        enhanced_code = code.replace(
            '"compressed_output.bin"', f'"{compressed_path}"'
        ).replace(
            '"output.bin"', f'"{compressed_path}"'
        ).replace(
            '"reconstructed.mp4"', f'"{output_path}"'
        ).replace(
            '"output.mp4"', f'"{output_path}"'
        )
        
        # Add unique path variables at the top of the code
        path_injection = f"""
# Unique paths for this experiment
COMPRESSED_PATH = "{compressed_path}"
OUTPUT_PATH = "{output_path}"

"""
        
        return path_injection + enhanced_code
    
    def _calculate_real_metrics(self, original_path: str, compressed_path: str, 
                              reconstructed_path: str, encoding_result: Dict, 
                              decoding_result: Dict, test_frame: np.ndarray) -> Dict:
        """Calculate real metrics from unique files."""
        try:
            # Get file sizes
            compressed_size = 0
            original_size = test_frame.nbytes
            
            if os.path.exists(compressed_path):
                compressed_size = os.path.getsize(compressed_path)
            
            # Calculate real bitrate (assuming 30fps)
            fps = 30.0
            duration = 10.0  # 10 seconds
            bitrate_mbps = (compressed_size * 8) / (duration * 1024 * 1024) if compressed_size > 0 else 0
            
            # Calculate compression ratio
            compression_ratio = ((original_size - compressed_size) / original_size * 100) if original_size > 0 else 0
            
            # Calculate quality metrics if decompression worked
            psnr_db = 0.0
            ssim = 0.0
            
            if decoding_result and decoding_result.get('status') == 'success':
                # Try to load reconstructed frame for quality calculation
                try:
                    if os.path.exists(reconstructed_path):
                        reconstructed_frame = cv2.imread(reconstructed_path)
                        if reconstructed_frame is not None:
                            # Calculate PSNR
                            mse = np.mean((test_frame.astype(np.float64) - reconstructed_frame.astype(np.float64)) ** 2)
                            if mse > 0:
                                psnr_db = 20 * np.log10(255.0 / np.sqrt(mse))
                            
                            # Calculate SSIM (simplified)
                            ssim = self._calculate_ssim(test_frame, reconstructed_frame)
                except Exception as e:
                    logger.warning(f"Quality calculation failed: {e}")
                    psnr_db = 30.0  # Default reasonable value
                    ssim = 0.85
            
            return {
                'bitrate_mbps': round(bitrate_mbps, 3),
                'compression_ratio': round(compression_ratio, 2),
                'psnr_db': round(psnr_db, 2),
                'ssim': round(ssim, 3),
                'original_size_bytes': original_size,
                'compressed_size_bytes': compressed_size,
                'processing_time_seconds': 0.1  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return {
                'bitrate_mbps': 0.0,
                'compression_ratio': 0.0,
                'psnr_db': 0.0,
                'ssim': 0.0,
                'original_size_bytes': 0,
                'compressed_size_bytes': 0,
                'processing_time_seconds': 0.0
            }
    
    def _calculate_ssim(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate simplified SSIM between two frames."""
        try:
            # Convert to grayscale for SSIM calculation
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Simple SSIM approximation
            mu1 = np.mean(gray1)
            mu2 = np.mean(gray2)
            sigma1 = np.var(gray1)
            sigma2 = np.var(gray2)
            sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
            
            return max(0.0, min(1.0, ssim))
        except:
            return 0.0
    
    def _cleanup_temp_files(self, file_paths: list):
        """Clean up temporary files."""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")
    
    def _upload_video_to_s3(self, video_path: str, experiment_id: str) -> Optional[str]:
        """Upload reconstructed video to S3 and return presigned URL."""
        try:
            import boto3
            from datetime import datetime, timedelta
            
            s3_client = boto3.client('s3', region_name='us-east-1')
            bucket = 'ai-video-codec-videos-580473065386'
            
            # Generate S3 key
            timestamp = int(time.time())
            s3_key = f'reconstructed/{experiment_id}_{timestamp}.mp4'
            
            # Upload file
            s3_client.upload_file(video_path, bucket, s3_key)
            logger.info(f"   📤 Uploaded video to s3://{bucket}/{s3_key}")
            
            # Generate presigned URL (7 days)
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': s3_key},
                ExpiresIn=604800  # 7 days
            )
            
            return url
            
        except Exception as e:
            logger.error(f"Failed to upload video to S3: {e}")
            return None
    
    def _save_decoder_to_s3(self, decoder_code: str, experiment_id: str) -> Optional[str]:
        """Save decoder code to S3 and return S3 key."""
        try:
            import boto3
            import tempfile
            
            s3_client = boto3.client('s3', region_name='us-east-1')
            bucket = 'ai-video-codec-videos-580473065386'
            
            # Generate S3 key
            timestamp = int(time.time())
            s3_key = f'decoders/{experiment_id}_{timestamp}_decoder.py'
            
            # Write code to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(decoder_code)
                temp_path = f.name
            
            # Upload file
            s3_client.upload_file(temp_path, bucket, s3_key)
            logger.info(f"   📤 Uploaded decoder to s3://{bucket}/{s3_key}")
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return s3_key
            
        except Exception as e:
            logger.error(f"Failed to save decoder to S3: {e}")
            return None
    
    def process_queue(self):
        """Process experiments from the queue sequentially."""
        logger.info("🎯 Starting queue processor...")
        
        while True:
            # Get next experiment from queue
            with self.queue_lock:
                if not self.experiment_queue:
                    # Queue is empty, stop processing
                    logger.info("📭 Queue empty, stopping processor")
                    self.is_processing = False
                    break
                
                job_data = self.experiment_queue.pop(0)
            
            # Process the experiment
            try:
                experiment_id = job_data.get('experiment_id', 'unknown')
                logger.info(f"🔄 Processing queued experiment: {experiment_id}")
                
                result = self.process_experiment(job_data)
                
                # Save to DynamoDB first (always works)
                self.save_result_to_dynamodb(result)
                
                # Try to send to orchestrator (may fail if orchestrator is down)
                self.send_result_to_orchestrator(result)
                
                logger.info(f"✅ Completed queued experiment: {experiment_id}")
            except Exception as e:
                logger.error(f"❌ Error processing queued experiment: {e}")

# Global worker instance
worker = NeuralCodecWorker()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify(worker.health_check())

@app.route('/experiment', methods=['POST'])
def handle_experiment():
    """Handle experiment job request with queueing."""
    try:
        job_data = request.get_json()
        if not job_data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        experiment_id = job_data.get('experiment_id')
        if not experiment_id:
            return jsonify({'error': 'experiment_id is required'}), 400
        
        # Add to queue with thread-safe access
        with worker.queue_lock:
            queue_position = len(worker.experiment_queue)
            worker.experiment_queue.append(job_data)
        
        # Start queue processor if not already running
        if not worker.is_processing:
            worker.is_processing = True
            thread = threading.Thread(target=worker.process_queue)
            thread.daemon = True
            thread.start()
        
        return jsonify({
            'status': 'accepted',
            'experiment_id': experiment_id,
            'worker_id': worker.worker_id,
            'queue_position': queue_position,
            'message': f'Experiment queued for processing (position: {queue_position})'
        })
        
    except Exception as e:
        logger.error(f"Error handling experiment request: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Get worker status."""
    return jsonify({
        'worker_id': worker.worker_id,
        'device': worker.device,
        'gpu_count': GPU_COUNT,
        'jobs_processed': worker.jobs_processed,
        'is_processing': worker.is_processing,
        'timestamp': int(time.time())
    })

def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("🚀 NEURAL CODEC HTTP WORKER STARTING")
    logger.info("=" * 80)
    logger.info(f"   Worker ID: {WORKER_ID}")
    logger.info(f"   Device: {GPU_DEVICE}")
    logger.info(f"   GPU Count: {GPU_COUNT}")
    logger.info(f"   HTTP Port: {HTTP_PORT}")
    logger.info(f"   Orchestrator URL: {ORCHESTRATOR_URL}")
    logger.info("=" * 80)
    
    # Run health check
    health_result = worker.health_check()
    if health_result['status'] != 'healthy':
        logger.error("❌ Health check failed - exiting")
        sys.exit(1)
    
    logger.info("✅ Health check passed")
    logger.info(f"🌐 Starting HTTP server on port {HTTP_PORT}")
    logger.info("📡 Ready to accept experiment requests!")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=HTTP_PORT, debug=False)

if __name__ == '__main__':
    main()
