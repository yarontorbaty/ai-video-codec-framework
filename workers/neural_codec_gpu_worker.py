#!/usr/bin/env python3
"""
Neural Codec GPU Worker
Executes two-agent neural video codec experiments on GPU hardware.
Polls SQS for jobs and processes encoding/decoding with quality measurement.
"""

import os
import sys
import json
import time
import logging
import traceback
import boto3
from datetime import datetime
from typing import Dict, Optional
import socket
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS clients
sqs = boto3.client('sqs', region_name='us-east-1')
s3 = boto3.client('s3', region_name='us-east-1')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

# Configuration
TRAINING_QUEUE_URL = os.environ.get(
    'TRAINING_QUEUE_URL',
    'https://sqs.us-east-1.amazonaws.com/580473065386/ai-video-codec-training-queue'
)
VIDEO_BUCKET = 'ai-video-codec-videos-580473065386'
EXPERIMENTS_TABLE = 'ai-video-codec-experiments'
WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"

# GPU detection
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import cv2
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
    from skimage.metrics import structural_similarity as ssim_metric
    
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_DEVICE = torch.device("cuda" if GPU_AVAILABLE else "cpu")
    
    if GPU_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"🎮 GPU Detected: {GPU_NAME} ({GPU_MEMORY:.1f} GB)")
    else:
        logger.warning("⚠️  No GPU detected - will use CPU")
        GPU_NAME = "CPU"
        GPU_MEMORY = 0
        
except ImportError as e:
    logger.error(f"❌ Failed to import required libraries: {e}")
    sys.exit(1)


class NeuralCodecExecutor:
    """Executes two-agent neural codec experiments on GPU."""
    
    def __init__(self):
        self.device = GPU_DEVICE
        self.gpu_available = GPU_AVAILABLE
        
    def load_video_from_s3(self, s3_path: str, duration: float, fps: float) -> torch.Tensor:
        """
        Load video from S3 and convert to tensor.
        
        Args:
            s3_path: S3 path (s3://bucket/key)
            duration: Duration in seconds to load
            fps: Frames per second
            
        Returns:
            Video tensor [1, T, C, H, W] normalized to [0, 1]
        """
        logger.info(f"  📥 Loading video from {s3_path}")
        
        # Parse S3 path
        parts = s3_path.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1]
        
        # Download to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
            s3.download_file(bucket, key, tmp_path)
        
        try:
            # Load video
            cap = cv2.VideoCapture(tmp_path)
            
            num_frames = int(duration * fps)
            frames = []
            
            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize to 1920x1080 if needed
                if frame.shape[:2] != (1080, 1920):
                    frame = cv2.resize(frame, (1920, 1080))
                
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                
                frames.append(frame)
            
            cap.release()
            
            # Convert to tensor [1, T, C, H, W]
            frames = np.stack(frames, axis=0)  # [T, H, W, C]
            frames = frames.transpose(0, 3, 1, 2)  # [T, C, H, W]
            frames_tensor = torch.from_numpy(frames).unsqueeze(0)  # [1, T, C, H, W]
            
            logger.info(f"  ✅ Loaded {len(frames)} frames at {frames_tensor.shape[3]}x{frames_tensor.shape[4]}")
            
            return frames_tensor
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def execute_encoding_agent(
        self,
        frames: torch.Tensor,
        encoding_code: str,
        config: Dict
    ) -> Dict:
        """
        Execute encoding agent code to compress video.
        
        Args:
            frames: Video frames [1, T, C, H, W]
            encoding_code: Python code for EncodingAgent
            config: Configuration dict
            
        Returns:
            Compressed data dict
        """
        logger.info(f"  🗜️  Executing encoding agent...")
        
        # Create safe execution environment
        exec_globals = {
            'torch': torch,
            'nn': nn,
            'F': F,
            'np': np,
            'cv2': cv2,
            'Dict': Dict,
            'List': list,
            'Tuple': tuple,
            'Optional': Optional,
            'logging': logging,
            'logger': logger
        }
        
        try:
            # Execute encoding agent code
            exec(encoding_code, exec_globals)
            
            # Get compress function
            if 'compress_video_tensor' in exec_globals:
                compress_fn = exec_globals['compress_video_tensor']
            else:
                raise ValueError("encoding_code must define compress_video_tensor() function")
            
            # Compress video
            compress_result = compress_fn(
                frames=frames,
                config=config,
                device=str(self.device)
            )
            
            logger.info(f"  ✅ Encoding complete")
            logger.info(f"     Bitrate: {compress_result['bitrate_mbps']:.4f} Mbps")
            logger.info(f"     Strategy: {compress_result['strategy']}")
            
            return compress_result
            
        except Exception as e:
            logger.error(f"  ❌ Encoding failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def execute_decoding_agent(
        self,
        compressed_data: Dict,
        decoding_code: str,
        config: Dict
    ) -> torch.Tensor:
        """
        Execute decoding agent code to reconstruct video.
        
        Args:
            compressed_data: Compressed representation from encoding
            decoding_code: Python code for DecodingAgent
            config: Configuration dict
            
        Returns:
            Reconstructed frames [1, T, C, H, W]
        """
        logger.info(f"  🔄 Executing decoding agent...")
        
        # Create safe execution environment
        exec_globals = {
            'torch': torch,
            'nn': nn,
            'F': F,
            'np': np,
            'cv2': cv2,
            'Dict': Dict,
            'List': list,
            'Tuple': tuple,
            'Optional': Optional,
            'logging': logging,
            'logger': logger
        }
        
        try:
            # Execute decoding agent code
            exec(decoding_code, exec_globals)
            
            # Get decompress function
            if 'decompress_video_tensor' in exec_globals:
                decompress_fn = exec_globals['decompress_video_tensor']
            else:
                raise ValueError("decoding_code must define decompress_video_tensor() function")
            
            # Decompress video
            decompress_result = decompress_fn(
                compressed_data=compressed_data['compressed_data'],
                config=config,
                device=str(self.device)
            )
            
            reconstructed = decompress_result['reconstructed_frames']
            
            logger.info(f"  ✅ Decoding complete")
            logger.info(f"     Decode FPS: {decompress_result['decode_fps']:.1f}")
            logger.info(f"     TOPS/frame: {decompress_result['tops_per_frame']:.4f}")
            
            return reconstructed, decompress_result
            
        except Exception as e:
            logger.error(f"  ❌ Decoding failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def calculate_quality_metrics(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> Dict:
        """
        Calculate quality metrics (PSNR, SSIM) between original and reconstructed.
        
        Args:
            original: Original frames [1, T, C, H, W]
            reconstructed: Reconstructed frames [1, T, C, H, W]
            
        Returns:
            Quality metrics dict
        """
        logger.info(f"  📊 Calculating quality metrics...")
        
        # Convert to numpy [T, H, W, C] in range [0, 1]
        orig_np = original[0].cpu().numpy().transpose(0, 2, 3, 1)
        recon_np = reconstructed[0].cpu().numpy().transpose(0, 2, 3, 1)
        
        # Calculate per-frame metrics
        psnr_values = []
        ssim_values = []
        
        for i in range(orig_np.shape[0]):
            # PSNR
            psnr = psnr_metric(orig_np[i], recon_np[i], data_range=1.0)
            psnr_values.append(psnr)
            
            # SSIM
            ssim = ssim_metric(
                orig_np[i],
                recon_np[i],
                data_range=1.0,
                channel_axis=2
            )
            ssim_values.append(ssim)
        
        # Average metrics
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        
        logger.info(f"  ✅ Quality metrics:")
        logger.info(f"     PSNR: {avg_psnr:.2f} dB")
        logger.info(f"     SSIM: {avg_ssim:.4f}")
        
        return {
            'psnr_db': float(avg_psnr),
            'ssim': float(avg_ssim),
            'psnr_per_frame': [float(x) for x in psnr_values],
            'ssim_per_frame': [float(x) for x in ssim_values]
        }
    
    def execute_experiment(self, job_data: Dict) -> Dict:
        """
        Execute complete two-agent neural codec experiment.
        
        Args:
            job_data: Job configuration from SQS
            
        Returns:
            Results dict with metrics
        """
        experiment_id = job_data['experiment_id']
        timestamp = job_data['timestamp']
        
        logger.info(f"🔬 Executing experiment: {experiment_id}")
        logger.info(f"   Device: {self.device}")
        
        start_time = time.time()
        
        try:
            # Extract job data
            encoding_code = job_data['encoding_agent_code']
            decoding_code = job_data['decoding_agent_code']
            config = job_data['config']
            video_path = config['video_path']
            duration = config['duration']
            fps = config['fps']
            
            # Step 1: Load video
            frames = self.load_video_from_s3(video_path, duration, fps)
            frames = frames.to(self.device)
            
            # Step 2: Encode video
            compress_result = self.execute_encoding_agent(frames, encoding_code, config)
            
            # Step 3: Decode video
            reconstructed, decompress_result = self.execute_decoding_agent(
                compress_result,
                decoding_code,
                config
            )
            
            # Step 4: Calculate quality
            quality_metrics = self.calculate_quality_metrics(frames, reconstructed)
            
            # Compile results
            elapsed_time = time.time() - start_time
            
            results = {
                'success': True,
                'experiment_id': experiment_id,
                'bitrate_mbps': compress_result['bitrate_mbps'],
                'compression_ratio': compress_result['compression_ratio'],
                'compressed_size_mb': compress_result['compressed_size_mb'],
                'strategy': compress_result['strategy'],
                'psnr_db': quality_metrics['psnr_db'],
                'ssim': quality_metrics['ssim'],
                'decode_fps': decompress_result['decode_fps'],
                'tops_per_frame': decompress_result['tops_per_frame'],
                'tops_at_30fps': decompress_result['tops_at_30fps'],
                'execution_time_seconds': elapsed_time,
                'worker_id': WORKER_ID,
                'device': str(self.device),
                'gpu_name': GPU_NAME
            }
            
            logger.info(f"✅ Experiment completed in {elapsed_time:.1f}s")
            logger.info(f"   Bitrate: {results['bitrate_mbps']:.4f} Mbps")
            logger.info(f"   PSNR: {results['psnr_db']:.2f} dB")
            logger.info(f"   SSIM: {results['ssim']:.4f}")
            logger.info(f"   TOPS: {results['tops_per_frame']:.4f} per frame")
            
            return results
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"Experiment execution error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'experiment_id': experiment_id,
                'error': error_msg,
                'traceback': traceback.format_exc(),
                'execution_time_seconds': elapsed_time,
                'worker_id': WORKER_ID,
                'device': str(self.device)
            }
    
    def update_experiment_status(self, experiment_id: str, timestamp: int, status: str, results: Dict):
        """Update experiment status in DynamoDB."""
        try:
            from decimal import Decimal
            
            def convert_floats(obj):
                if isinstance(obj, dict):
                    return {k: convert_floats(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_floats(item) for item in obj]
                elif isinstance(obj, float):
                    return Decimal(str(obj))
                return obj
            
            table = dynamodb.Table(EXPERIMENTS_TABLE)
            
            # Update experiment with GPU results
            update_data = {
                'gpu_status': status,
                'gpu_results': results,
                'gpu_worker_id': WORKER_ID,
                'gpu_completed_at': datetime.utcnow().isoformat() + 'Z'
            }
            
            # Get current item to preserve other fields
            response = table.get_item(
                Key={
                    'experiment_id': experiment_id,
                    'timestamp': timestamp
                }
            )
            
            item = response.get('Item', {})
            item.update(convert_floats(update_data))
            
            # Put updated item
            table.put_item(Item=item)
            
            logger.info(f"✅ Updated experiment {experiment_id} status to {status}")
            
        except Exception as e:
            logger.error(f"Failed to update experiment status: {e}")
            logger.error(traceback.format_exc())


class NeuralCodecWorker:
    """Main worker that polls SQS and executes neural codec experiments."""
    
    def __init__(self):
        self.executor = NeuralCodecExecutor()
        self.running = True
        self.jobs_processed = 0
        
    def poll_queue(self) -> Optional[Dict]:
        """Poll SQS queue for new experiment jobs."""
        try:
            response = sqs.receive_message(
                QueueUrl=TRAINING_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,  # Long polling
                VisibilityTimeout=3600  # 1 hour for processing
            )
            
            messages = response.get('Messages', [])
            if not messages:
                return None
            
            message = messages[0]
            receipt_handle = message['ReceiptHandle']
            body = json.loads(message['Body'])
            
            return {
                'receipt_handle': receipt_handle,
                'data': body
            }
            
        except Exception as e:
            logger.error(f"Error polling queue: {e}")
            return None
    
    def delete_message(self, receipt_handle: str):
        """Delete message from queue after successful processing."""
        try:
            sqs.delete_message(
                QueueUrl=TRAINING_QUEUE_URL,
                ReceiptHandle=receipt_handle
            )
            logger.info("✅ Message deleted from queue")
        except Exception as e:
            logger.error(f"Failed to delete message: {e}")
    
    def run(self):
        """Main worker loop."""
        logger.info("=" * 80)
        logger.info("🚀 NEURAL CODEC GPU WORKER STARTED")
        logger.info(f"   Worker ID: {WORKER_ID}")
        logger.info(f"   GPU: {GPU_NAME}")
        logger.info(f"   Device: {GPU_DEVICE}")
        logger.info(f"   Queue: {TRAINING_QUEUE_URL}")
        logger.info("=" * 80)
        
        while self.running:
            try:
                # Poll for jobs
                logger.info("\n📥 Polling for neural codec experiments...")
                job = self.poll_queue()
                
                if not job:
                    logger.info("   No jobs available, waiting...")
                    continue
                
                # Process job
                job_data = job['data']
                experiment_id = job_data.get('experiment_id', 'unknown')
                timestamp = job_data.get('timestamp', int(time.time()))
                
                logger.info(f"\n🎯 Received job: {experiment_id}")
                
                # Update status to processing
                self.executor.update_experiment_status(
                    experiment_id,
                    timestamp,
                    'processing',
                    {'worker_id': WORKER_ID, 'started_at': datetime.utcnow().isoformat()}
                )
                
                # Execute experiment
                results = self.executor.execute_experiment(job_data)
                
                # Update final status
                final_status = 'completed' if results['success'] else 'failed'
                self.executor.update_experiment_status(
                    experiment_id,
                    timestamp,
                    final_status,
                    results
                )
                
                # Delete message from queue
                self.delete_message(job['receipt_handle'])
                
                self.jobs_processed += 1
                logger.info(f"\n✅ Job completed. Total processed: {self.jobs_processed}")
                
            except KeyboardInterrupt:
                logger.info("\n🛑 Shutting down worker...")
                self.running = False
                break
                
            except Exception as e:
                logger.error(f"\n❌ Worker error: {e}")
                logger.error(traceback.format_exc())
                time.sleep(5)  # Brief pause before retry


def main():
    """Main entry point."""
    # Health check
    logger.info("Running health check...")
    
    if not GPU_AVAILABLE:
        logger.warning("⚠️  WARNING: No GPU available - performance will be degraded")
    
    try:
        # Test SQS connectivity
        sqs.get_queue_attributes(
            QueueUrl=TRAINING_QUEUE_URL,
            AttributeNames=['ApproximateNumberOfMessages']
        )
        logger.info("✅ SQS connection OK")
    except Exception as e:
        logger.error(f"❌ Cannot connect to SQS: {e}")
        logger.error("   Worker may not receive jobs")
    
    # Start worker
    worker = NeuralCodecWorker()
    worker.run()


if __name__ == '__main__':
    main()


