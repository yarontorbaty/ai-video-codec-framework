#!/usr/bin/env python3
"""
GPU Training Worker
Polls SQS queue for experiment jobs and executes them on GPU hardware.
Supports both training and inference workloads with PyTorch.
"""

import os
import sys
import json
import time
import logging
import traceback
import boto3
from datetime import datetime
from typing import Dict, Optional, Any
import socket

# Add parent directory to path for imports
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
TRAINING_QUEUE_URL = os.environ.get('TRAINING_QUEUE_URL', 'https://sqs.us-east-1.amazonaws.com/580473065386/ai-video-codec-training-queue')
EXPERIMENTS_TABLE = 'ai-video-codec-experiments'
WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"

# GPU detection
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_DEVICE = torch.device("cuda" if GPU_AVAILABLE else "cpu")
    if GPU_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        logger.info(f"🎮 GPU Detected: {GPU_NAME} ({GPU_MEMORY:.1f} GB)")
    else:
        logger.warning("⚠️  No GPU detected - will use CPU")
        GPU_NAME = "CPU"
        GPU_MEMORY = 0
except ImportError:
    logger.error("❌ PyTorch not installed!")
    GPU_AVAILABLE = False
    GPU_DEVICE = "cpu"
    GPU_NAME = "CPU"
    GPU_MEMORY = 0


class GPUExperimentExecutor:
    """Executes experiment code on GPU hardware."""
    
    def __init__(self):
        self.device = GPU_DEVICE
        self.gpu_available = GPU_AVAILABLE
        
    def execute_experiment(self, experiment_data: Dict) -> Dict:
        """
        Execute an experiment with GPU acceleration.
        
        Args:
            experiment_data: Experiment configuration from SQS
            
        Returns:
            Results dictionary with metrics and status
        """
        experiment_id = experiment_data.get('experiment_id')
        code = experiment_data.get('code')
        config = experiment_data.get('config', {})
        
        logger.info(f"🔬 Executing experiment: {experiment_id}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Code length: {len(code)} chars")
        
        start_time = time.time()
        
        try:
            # Import the code sandbox for safe execution
            from utils.code_sandbox import CodeSandbox
            
            sandbox = CodeSandbox()
            
            # Execute code in sandbox with GPU access
            success, metrics, error_msg = sandbox.execute_compression_code(
                code=code,
                duration=config.get('duration', 10.0),
                fps=config.get('fps', 30.0),
                resolution=tuple(config.get('resolution', [1920, 1080]))
            )
            
            elapsed_time = time.time() - start_time
            
            if success:
                logger.info(f"✅ Experiment completed successfully in {elapsed_time:.1f}s")
                logger.info(f"   Bitrate: {metrics.get('bitrate_mbps', 0):.4f} Mbps")
                
                # Add GPU info to metrics
                metrics['execution_device'] = str(self.device)
                metrics['gpu_name'] = GPU_NAME
                metrics['gpu_memory_gb'] = GPU_MEMORY
                metrics['execution_time_seconds'] = elapsed_time
                metrics['worker_id'] = WORKER_ID
                
                return {
                    'success': True,
                    'experiment_id': experiment_id,
                    'metrics': metrics,
                    'elapsed_time': elapsed_time,
                    'device': str(self.device)
                }
            else:
                logger.error(f"❌ Experiment failed: {error_msg}")
                return {
                    'success': False,
                    'experiment_id': experiment_id,
                    'error': error_msg,
                    'elapsed_time': elapsed_time,
                    'device': str(self.device)
                }
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"GPU execution error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'experiment_id': experiment_id,
                'error': error_msg,
                'traceback': traceback.format_exc(),
                'elapsed_time': elapsed_time,
                'device': str(self.device)
            }
    
    def update_experiment_status(self, experiment_id: str, status: str, results: Dict):
        """Update experiment status in DynamoDB."""
        try:
            table = dynamodb.Table(EXPERIMENTS_TABLE)
            
            # Get current experiment data
            response = table.query(
                KeyConditionExpression='experiment_id = :id',
                ExpressionAttributeValues={':id': experiment_id}
            )
            
            if not response.get('Items'):
                logger.error(f"Experiment {experiment_id} not found in DynamoDB")
                return
            
            item = response['Items'][0]
            timestamp = item['timestamp']
            
            # Update with results
            update_expr = "SET #status = :status, gpu_execution_results = :results"
            expr_attr_names = {'#status': 'status'}
            expr_attr_values = {
                ':status': status,
                ':results': json.dumps(results)
            }
            
            table.update_item(
                Key={
                    'experiment_id': experiment_id,
                    'timestamp': timestamp
                },
                UpdateExpression=update_expr,
                ExpressionAttributeNames=expr_attr_names,
                ExpressionAttributeValues=expr_attr_values
            )
            
            logger.info(f"✅ Updated experiment {experiment_id} status to {status}")
            
        except Exception as e:
            logger.error(f"Failed to update experiment status: {e}")


class TrainingWorker:
    """Main training worker that polls SQS and executes jobs."""
    
    def __init__(self):
        self.executor = GPUExperimentExecutor()
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
        logger.info("🚀 GPU Training Worker Started")
        logger.info(f"   Worker ID: {WORKER_ID}")
        logger.info(f"   GPU: {GPU_NAME}")
        logger.info(f"   Device: {GPU_DEVICE}")
        logger.info(f"   Queue: {TRAINING_QUEUE_URL}")
        logger.info("=" * 80)
        
        while self.running:
            try:
                # Poll for jobs
                logger.info("📥 Polling for experiment jobs...")
                job = self.poll_queue()
                
                if not job:
                    logger.info("   No jobs available, waiting...")
                    continue
                
                # Process job
                experiment_data = job['data']
                experiment_id = experiment_data.get('experiment_id', 'unknown')
                
                logger.info(f"🎯 Received job: {experiment_id}")
                
                # Update status to processing
                self.executor.update_experiment_status(
                    experiment_id,
                    'processing_on_gpu',
                    {'worker_id': WORKER_ID, 'started_at': datetime.utcnow().isoformat()}
                )
                
                # Execute experiment
                results = self.executor.execute_experiment(experiment_data)
                
                # Update final status
                final_status = 'completed' if results['success'] else 'failed'
                self.executor.update_experiment_status(
                    experiment_id,
                    final_status,
                    results
                )
                
                # Delete message from queue
                self.delete_message(job['receipt_handle'])
                
                self.jobs_processed += 1
                logger.info(f"✅ Job completed. Total processed: {self.jobs_processed}")
                
            except KeyboardInterrupt:
                logger.info("🛑 Shutting down worker...")
                self.running = False
                break
                
            except Exception as e:
                logger.error(f"❌ Worker error: {e}")
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
    worker = TrainingWorker()
    worker.run()


if __name__ == '__main__':
    main()

