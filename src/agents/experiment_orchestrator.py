#!/usr/bin/env python3
"""
Experiment Orchestrator
Coordinates AI codec experiments, manages resources, and reports results.
"""

import os
import sys
import json
import time
import logging
import threading
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yaml
import psutil
import boto3
from botocore.exceptions import ClientError

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ai_codec_agent import AICodecAgent
from agents.procedural_generator import ProceduralCompressionAgent
from utils.aws_utils import AWSUtils
from utils.metrics import MetricsCalculator
from utils.video_utils import VideoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentOrchestrator:
    """Orchestrates AI codec experiments with resource management."""
    
    def __init__(self, config_path: str = "config/ai_codec_config.yaml"):
        self.config = self._load_config(config_path)
        self.aws_utils = AWSUtils()
        self.metrics_calc = MetricsCalculator()
        self.video_processor = VideoProcessor()
        
        # Initialize agents
        self.ai_agent = AICodecAgent(config_path)
        self.procedural_agent = ProceduralCompressionAgent()
        
        # Experiment state
        self.current_experiment = None
        self.experiment_history = []
        self.resource_monitor = ResourceMonitor()
        
        # Cost tracking
        self.cost_tracker = CostTracker()
        
        # Create necessary directories
        self._create_directories()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'experiment': {
                'max_concurrent': 1,
                'timeout_hours': 2,
                'retry_attempts': 3
            },
            'cost_management': {
                'max_cost_per_experiment': 50.0,
                'max_daily_cost': 200.0
            }
        }
    
    def _create_directories(self):
        """Create necessary directories."""
        dirs = [
            'data/source',
            'data/hevc', 
            'data/compressed',
            'data/results',
            'logs',
            'models',
            'checkpoints'
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def start_experiment(self, experiment_id: Optional[str] = None) -> str:
        """Start a new AI codec experiment."""
        if experiment_id is None:
            experiment_id = f"exp_{int(time.time())}"
        
        logger.info(f"Starting experiment: {experiment_id}")
        
        # Check cost limits
        if not self.cost_tracker.can_start_experiment():
            raise RuntimeError("Cost limit exceeded, cannot start experiment")
        
        # Create experiment record
        experiment_config = {
            'experiment_id': experiment_id,
            'start_time': datetime.utcnow().isoformat(),
            'status': 'running',
            'config': self.config
        }
        
        self.aws_utils.create_experiment_record(experiment_id, experiment_config)
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            # Run the experiment
            results = self._run_experiment(experiment_id)
            
            # Update experiment status
            self.aws_utils.update_experiment_status(
                experiment_id, 'completed', results
            )
            
            logger.info(f"Experiment {experiment_id} completed successfully")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}")
            self.aws_utils.update_experiment_status(
                experiment_id, 'failed', {'error': str(e)}
            )
            raise
        
        finally:
            # Stop resource monitoring
            self.resource_monitor.stop_monitoring()
    
    def _run_experiment(self, experiment_id: str) -> Dict:
        """Run the actual AI codec experiment."""
        logger.info(f"Running experiment: {experiment_id}")
        
        # Download test data
        source_path, hevc_path = self._download_test_data()
        
        # Run AI codec experiment
        logger.info("Starting AI codec experiment...")
        ai_results = self.ai_agent.run_experiment()
        
        # Run procedural generation experiment
        logger.info("Starting procedural generation experiment...")
        procedural_results = self._run_procedural_experiment(source_path)
        
        # Run hybrid experiment
        logger.info("Starting hybrid experiment...")
        hybrid_results = self._run_hybrid_experiment(source_path)
        
        # Compare all approaches
        comparison_results = self._compare_approaches(
            ai_results, procedural_results, hybrid_results
        )
        
        # Calculate final metrics
        final_results = {
            'experiment_id': experiment_id,
            'timestamp': datetime.utcnow().isoformat(),
            'ai_codec_results': ai_results,
            'procedural_results': procedural_results,
            'hybrid_results': hybrid_results,
            'comparison': comparison_results,
            'resource_usage': self.resource_monitor.get_usage_stats(),
            'cost': self.cost_tracker.get_current_cost()
        }
        
        # Save results
        self._save_results(experiment_id, final_results)
        
        # Upload results to S3
        self._upload_results(experiment_id, final_results)
        
        return final_results
    
    def _download_test_data(self) -> tuple:
        """Download test data from S3."""
        logger.info("Downloading test data from S3...")
        
        # Download source video
        source_path = "data/source/SOURCE_HD_RAW.mp4"
        self.aws_utils.download_from_s3(
            bucket=self.config['aws']['s3_bucket'],
            key=self.config['experiment']['source_video'],
            local_path=source_path
        )
        
        # Download HEVC reference
        hevc_path = "data/hevc/HEVC_HD_10Mbps.mp4"
        self.aws_utils.download_from_s3(
            bucket=self.config['aws']['s3_bucket'],
            key=self.config['experiment']['hevc_reference'],
            local_path=hevc_path
        )
        
        return source_path, hevc_path
    
    def _run_procedural_experiment(self, source_path: str) -> Dict:
        """Run procedural generation experiment."""
        logger.info("Running procedural generation experiment...")
        
        # Generate procedural video for comparison
        procedural_path = "data/compressed/procedural_output.mp4"
        procedural_results = self.procedural_agent.generate_procedural_video(
            procedural_path, duration=10.0, fps=30.0
        )
        
        # Calculate quality metrics
        quality_metrics = self.metrics_calc.calculate_all_metrics(
            source_path, procedural_path
        )
        
        return {
            'procedural_generation': procedural_results,
            'quality_metrics': quality_metrics
        }
    
    def _run_hybrid_experiment(self, source_path: str) -> Dict:
        """Run hybrid AI + procedural experiment."""
        logger.info("Running hybrid experiment...")
        
        # Use the hybrid compression method
        hybrid_path = "data/compressed/hybrid_output.mp4"
        hybrid_results = self.ai_agent.compress_video_hybrid(source_path, hybrid_path)
        
        return hybrid_results
    
    def _compare_approaches(self, ai_results: Dict, procedural_results: Dict, 
                          hybrid_results: Dict) -> Dict:
        """Compare all compression approaches."""
        logger.info("Comparing compression approaches...")
        
        # Extract key metrics
        approaches = {
            'ai_codec': {
                'bitrate_mbps': ai_results.get('ai_codec_metrics', {}).get('bitrate_mbps', 0),
                'psnr_db': ai_results.get('ai_codec_metrics', {}).get('psnr_db', 0),
                'compression_ratio': ai_results.get('ai_codec_metrics', {}).get('compression_ratio', 0)
            },
            'procedural': {
                'bitrate_mbps': procedural_results.get('procedural_generation', {}).get('bitrate_mbps', 0),
                'psnr_db': procedural_results.get('quality_metrics', {}).get('psnr_db', 0),
                'compression_ratio': procedural_results.get('quality_metrics', {}).get('compression_ratio', 0)
            },
            'hybrid': {
                'bitrate_mbps': hybrid_results.get('bitrate_mbps', 0),
                'psnr_db': hybrid_results.get('psnr_db', 0),
                'compression_ratio': hybrid_results.get('compression_ratio', 0)
            }
        }
        
        # Find best approach for each metric
        best_bitrate = min(approaches.items(), key=lambda x: x[1]['bitrate_mbps'])
        best_psnr = max(approaches.items(), key=lambda x: x[1]['psnr_db'])
        best_compression = min(approaches.items(), key=lambda x: x[1]['compression_ratio'])
        
        comparison = {
            'approaches': approaches,
            'best_bitrate': {
                'approach': best_bitrate[0],
                'value': best_bitrate[1]['bitrate_mbps']
            },
            'best_psnr': {
                'approach': best_psnr[0],
                'value': best_psnr[1]['psnr_db']
            },
            'best_compression': {
                'approach': best_compression[0],
                'value': best_compression[1]['compression_ratio']
            },
            'target_achievement': {
                'bitrate_target': any(r['bitrate_mbps'] < 1.0 for r in approaches.values()),
                'psnr_target': any(r['psnr_db'] > 35.0 for r in approaches.values()),
                'compression_target': any(r['compression_ratio'] < 0.1 for r in approaches.values())
            }
        }
        
        return comparison
    
    def _save_results(self, experiment_id: str, results: Dict):
        """Save experiment results locally."""
        results_path = f"data/results/{experiment_id}_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_path}")
    
    def _upload_results(self, experiment_id: str, results: Dict):
        """Upload results to S3."""
        results_path = f"data/results/{experiment_id}_results.json"
        
        self.aws_utils.upload_to_s3(
            local_path=results_path,
            bucket=self.config['aws']['s3_bucket'],
            key=f"results/{experiment_id}_results.json",
            metadata={
                'experiment_id': experiment_id,
                'timestamp': results['timestamp'],
                'type': 'experiment_results'
            }
        )
        
        logger.info(f"Results uploaded to S3: s3://{self.config['aws']['s3_bucket']}/results/{experiment_id}_results.json")
    
    def get_experiment_status(self, experiment_id: str) -> Optional[Dict]:
        """Get status of a specific experiment."""
        return self.aws_utils.get_experiment_status(experiment_id)
    
    def list_experiments(self) -> List[Dict]:
        """List all experiments."""
        # This would query DynamoDB for all experiments
        # For now, return local experiment history
        return self.experiment_history
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop a running experiment."""
        logger.info(f"Stopping experiment: {experiment_id}")
        
        # Update status in DynamoDB
        self.aws_utils.update_experiment_status(
            experiment_id, 'stopped', {'stopped_at': datetime.utcnow().isoformat()}
        )
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        return True


class ResourceMonitor:
    """Monitor system resources during experiments."""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.usage_stats = {
            'cpu_percent': [],
            'memory_percent': [],
            'gpu_usage': [],
            'disk_usage': []
        }
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Resource monitoring loop."""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.usage_stats['cpu_percent'].append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.usage_stats['memory_percent'].append(memory.percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.usage_stats['disk_usage'].append(disk.percent)
                
                # GPU usage (if available)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_usage = gpus[0].load * 100
                        self.usage_stats['gpu_usage'].append(gpu_usage)
                except ImportError:
                    pass
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(10)
    
    def get_usage_stats(self) -> Dict:
        """Get resource usage statistics."""
        if not self.usage_stats['cpu_percent']:
            return {}
        
        return {
            'avg_cpu_percent': sum(self.usage_stats['cpu_percent']) / len(self.usage_stats['cpu_percent']),
            'max_cpu_percent': max(self.usage_stats['cpu_percent']),
            'avg_memory_percent': sum(self.usage_stats['memory_percent']) / len(self.usage_stats['memory_percent']),
            'max_memory_percent': max(self.usage_stats['memory_percent']),
            'avg_disk_percent': sum(self.usage_stats['disk_usage']) / len(self.usage_stats['disk_usage']),
            'max_disk_percent': max(self.usage_stats['disk_usage']),
            'gpu_usage': self.usage_stats['gpu_usage'][-1] if self.usage_stats['gpu_usage'] else None
        }


class CostTracker:
    """Track experiment costs."""
    
    def __init__(self):
        self.current_cost = 0.0
        self.daily_cost = 0.0
        self.cost_history = []
    
    def can_start_experiment(self) -> bool:
        """Check if we can start a new experiment based on cost limits."""
        # This would check against AWS cost limits
        # For now, just return True
        return True
    
    def get_current_cost(self) -> Dict:
        """Get current cost information."""
        return {
            'current_cost': self.current_cost,
            'daily_cost': self.daily_cost,
            'cost_history': self.cost_history
        }


def main():
    """Main function for experiment orchestrator."""
    parser = argparse.ArgumentParser(description='AI Video Codec Experiment Orchestrator')
    parser.add_argument('--config', type=str, default='config/ai_codec_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--experiment-id', type=str, help='Experiment ID')
    parser.add_argument('--action', choices=['start', 'status', 'stop', 'list'], 
                       default='start', help='Action to perform')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = ExperimentOrchestrator(args.config)
    
    if args.action == 'start':
        # Start new experiment
        experiment_id = orchestrator.start_experiment(args.experiment_id)
        print(f"Started experiment: {experiment_id}")
        
    elif args.action == 'status':
        if not args.experiment_id:
            print("Experiment ID required for status check")
            return
        
        status = orchestrator.get_experiment_status(args.experiment_id)
        if status:
            print(f"Experiment {args.experiment_id} status: {status}")
        else:
            print(f"Experiment {args.experiment_id} not found")
            
    elif args.action == 'stop':
        if not args.experiment_id:
            print("Experiment ID required to stop experiment")
            return
        
        success = orchestrator.stop_experiment(args.experiment_id)
        if success:
            print(f"Experiment {args.experiment_id} stopped")
        else:
            print(f"Failed to stop experiment {args.experiment_id}")
            
    elif args.action == 'list':
        experiments = orchestrator.list_experiments()
        print(f"Found {len(experiments)} experiments:")
        for exp in experiments:
            print(f"  - {exp}")


if __name__ == "__main__":
    main()
