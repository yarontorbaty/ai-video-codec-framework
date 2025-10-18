#!/usr/bin/env python3
"""
GPU-First Orchestrator
Coordinates experiments but NEVER executes them locally.
All work is dispatched to GPU workers via SQS.
"""

import os
import sys
import json
import time
import logging
import boto3
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm_experiment_planner import LLMExperimentPlanner

logger = logging.getLogger(__name__)

# AWS Configuration
VIDEO_BUCKET = 'ai-video-codec-videos-580473065386'
TRAINING_QUEUE_URL = 'https://sqs.us-east-1.amazonaws.com/580473065386/ai-video-codec-training-queue'
EXPERIMENTS_TABLE = 'ai-video-codec-experiments'


class ExperimentPhase(Enum):
    """Phases of experiment orchestration"""
    DESIGN = "design"
    DISPATCH = "dispatch"
    WAITING_GPU = "waiting_gpu"
    ANALYSIS = "analysis"
    COMPLETE = "complete"
    FAILED = "failed"


class GPUFirstOrchestrator:
    """
    Orchestrator that coordinates experiment lifecycle but executes nothing locally.
    
    Workflow:
    1. Design: Analyze past results, generate neural architecture code
    2. Dispatch: Send experiment to GPU worker via SQS
    3. Wait: Poll for completion from GPU worker
    4. Analyze: Evaluate results and design next iteration
    """
    
    def __init__(self):
        """Initialize the GPU-first orchestrator."""
        self.planner = LLMExperimentPlanner()
        
        # AWS clients
        self.sqs = boto3.client('sqs', region_name='us-east-1')
        self.s3 = boto3.client('s3', region_name='us-east-1')
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        self.experiments_table = self.dynamodb.Table(EXPERIMENTS_TABLE)
        
        # State tracking
        self.current_phase = ExperimentPhase.DESIGN
        
        # Timeouts
        self.gpu_timeout_seconds = 1800  # 30 minutes max for GPU execution
        self.poll_interval_seconds = 10
    
    def run_experiment_cycle(self, iteration: int) -> Dict:
        """
        Run a complete experiment cycle.
        
        Args:
            iteration: Experiment iteration number
            
        Returns:
            Experiment results dict
        """
        experiment_id = f"gpu_exp_{int(time.time())}"
        start_time = time.time()
        timestamp = int(start_time)
        
        logger.info("=" * 80)
        logger.info(f"🚀 GPU-FIRST EXPERIMENT CYCLE {iteration}")
        logger.info(f"   ID: {experiment_id}")
        logger.info(f"   Timestamp: {timestamp}")
        logger.info("=" * 80)
        
        try:
            # ============================================================
            # PHASE 1: DESIGN (Orchestrator)
            # ============================================================
            self.current_phase = ExperimentPhase.DESIGN
            logger.info("\n📐 PHASE 1: DESIGN (Orchestrator)")
            logger.info("   Analyzing past experiments...")
            
            design_result = self._phase_design(experiment_id, timestamp)
            
            if not design_result['success']:
                return self._create_failure_result(experiment_id, "design_failed", design_result)
            
            logger.info(f"   ✅ Design complete")
            logger.info(f"   Strategy: {design_result.get('strategy', 'N/A')}")
            
            # ============================================================
            # PHASE 2: DISPATCH TO GPU (Orchestrator → SQS)
            # ============================================================
            self.current_phase = ExperimentPhase.DISPATCH
            logger.info("\n📤 PHASE 2: DISPATCH TO GPU")
            logger.info("   Sending experiment to GPU worker queue...")
            
            dispatch_result = self._phase_dispatch_to_gpu(experiment_id, timestamp, design_result)
            
            if not dispatch_result['success']:
                return self._create_failure_result(experiment_id, "dispatch_failed", dispatch_result)
            
            logger.info(f"   ✅ Dispatched to SQS")
            logger.info(f"   Message ID: {dispatch_result['message_id']}")
            
            # ============================================================
            # PHASE 3: WAIT FOR GPU (GPU Worker)
            # ============================================================
            self.current_phase = ExperimentPhase.WAITING_GPU
            logger.info("\n⏳ PHASE 3: WAITING FOR GPU WORKER")
            logger.info(f"   Polling for results (timeout: {self.gpu_timeout_seconds}s)...")
            
            gpu_result = self._phase_wait_for_gpu(experiment_id, timestamp)
            
            if not gpu_result['success']:
                return self._create_failure_result(experiment_id, "gpu_execution_failed", gpu_result)
            
            logger.info(f"   ✅ GPU execution complete")
            logger.info(f"   Bitrate: {gpu_result.get('bitrate_mbps', 0):.4f} Mbps")
            logger.info(f"   PSNR: {gpu_result.get('psnr_db', 0):.2f} dB")
            
            # ============================================================
            # PHASE 4: ANALYSIS (Orchestrator)
            # ============================================================
            self.current_phase = ExperimentPhase.ANALYSIS
            logger.info("\n📊 PHASE 4: ANALYSIS")
            logger.info("   Evaluating results against targets...")
            
            analysis_result = self._phase_analysis(experiment_id, timestamp, gpu_result)
            
            # ============================================================
            # PHASE 5: COMPLETE
            # ============================================================
            self.current_phase = ExperimentPhase.COMPLETE
            elapsed = time.time() - start_time
            
            logger.info("\n✅ EXPERIMENT CYCLE COMPLETE")
            logger.info(f"   Total time: {elapsed:.1f}s")
            logger.info(f"   Success: {analysis_result['target_achieved']}")
            
            return self._create_success_result(experiment_id, analysis_result)
            
        except Exception as e:
            logger.error(f"\n❌ UNEXPECTED ERROR: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._create_failure_result(experiment_id, "unexpected_error", {'error': str(e)})
    
    def _phase_design(self, experiment_id: str, timestamp: int) -> Dict:
        """
        Phase 1: Design experiment using LLM.
        
        This phase:
        - Analyzes past experiment results
        - Identifies patterns and bottlenecks
        - Generates PyTorch code for encoding and decoding agents
        - Selects compression strategy
        """
        try:
            # Update DynamoDB status
            self._update_experiment_status(experiment_id, timestamp, 'design', {
                'status': 'running',
                'phase': 'design'
            })
            
            # Fetch recent experiments for analysis
            recent_experiments = self.planner.analyze_recent_experiments(limit=10)
            
            logger.info(f"   Found {len(recent_experiments)} recent experiments")
            
            # Get LLM analysis and code generation
            llm_analysis = self.planner.get_llm_analysis(recent_experiments)
            
            if not llm_analysis:
                logger.error("   ❌ LLM analysis failed")
                return {
                    'success': False,
                    'error': 'LLM analysis unavailable'
                }
            
            # Extract generated code
            encoding_code = llm_analysis.get('encoding_agent_code')
            decoding_code = llm_analysis.get('decoding_agent_code')
            strategy = llm_analysis.get('compression_strategy', 'hybrid_semantic')
            
            if not encoding_code or not decoding_code:
                logger.error("   ❌ LLM did not generate both agent codes")
                return {
                    'success': False,
                    'error': 'Missing encoding or decoding agent code'
                }
            
            logger.info(f"   Generated encoding agent: {len(encoding_code)} chars")
            logger.info(f"   Generated decoding agent: {len(decoding_code)} chars")
            logger.info(f"   Strategy: {strategy}")
            
            # Write blog post with hypothesis
            self._write_blog_post_design(experiment_id, timestamp, llm_analysis)
            
            return {
                'success': True,
                'encoding_code': encoding_code,
                'decoding_code': decoding_code,
                'strategy': strategy,
                'llm_analysis': llm_analysis
            }
            
        except Exception as e:
            logger.error(f"   ❌ Design phase error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
    def _phase_dispatch_to_gpu(
        self,
        experiment_id: str,
        timestamp: int,
        design_result: Dict
    ) -> Dict:
        """
        Phase 2: Dispatch experiment to GPU worker via SQS.
        
        This phase:
        - Packages experiment configuration
        - Sends job to SQS queue
        - GPU worker will pick it up and execute
        """
        try:
            # Update status
            self._update_experiment_status(experiment_id, timestamp, 'dispatch', {
                'status': 'dispatching',
                'phase': 'dispatch'
            })
            
            # Prepare job payload
            job_payload = {
                'experiment_id': experiment_id,
                'timestamp': timestamp,
                'encoding_agent_code': design_result['encoding_code'],
                'decoding_agent_code': design_result['decoding_code'],
                'strategy': design_result['strategy'],
                'config': {
                    'video_path': f's3://{VIDEO_BUCKET}/test_data/SOURCE_HD_RAW.mp4',
                    'duration': 10.0,
                    'fps': 30.0,
                    'resolution': [1920, 1080],
                    'target_bitrate_mbps': 1.0,
                    'target_psnr_db': 35.0,
                    'target_ssim': 0.95,
                    'latent_dim': 512,
                    'description_dim': 256,
                    'i_frame_interval': 30,
                    'use_temporal_enhancement': True
                },
                'dispatched_at': datetime.utcnow().isoformat() + 'Z'
            }
            
            # Send to SQS
            response = self.sqs.send_message(
                QueueUrl=TRAINING_QUEUE_URL,
                MessageBody=json.dumps(job_payload),
                MessageAttributes={
                    'experiment_id': {
                        'StringValue': experiment_id,
                        'DataType': 'String'
                    },
                    'timestamp': {
                        'StringValue': str(timestamp),
                        'DataType': 'Number'
                    }
                }
            )
            
            message_id = response['MessageId']
            
            # Update status
            self._update_experiment_status(experiment_id, timestamp, 'waiting_gpu', {
                'status': 'waiting_for_gpu',
                'phase': 'waiting_gpu',
                'sqs_message_id': message_id
            })
            
            return {
                'success': True,
                'message_id': message_id
            }
            
        except Exception as e:
            logger.error(f"   ❌ Dispatch error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
    def _phase_wait_for_gpu(self, experiment_id: str, timestamp: int) -> Dict:
        """
        Phase 3: Wait for GPU worker to complete experiment.
        
        This phase:
        - Polls DynamoDB for results
        - GPU worker updates the experiment record when complete
        - Times out after gpu_timeout_seconds
        """
        start_wait = time.time()
        
        while (time.time() - start_wait) < self.gpu_timeout_seconds:
            # Poll DynamoDB
            try:
                response = self.experiments_table.get_item(
                    Key={
                        'experiment_id': experiment_id,
                        'timestamp': timestamp
                    }
                )
                
                item = response.get('Item')
                
                if item:
                    gpu_status = item.get('gpu_status')
                    
                    if gpu_status == 'completed':
                        # GPU worker completed successfully
                        gpu_results = item.get('gpu_results', {})
                        
                        # Parse if JSON string
                        if isinstance(gpu_results, str):
                            gpu_results = json.loads(gpu_results)
                        
                        logger.info(f"   ✅ GPU worker completed")
                        
                        return {
                            'success': True,
                            'bitrate_mbps': gpu_results.get('bitrate_mbps', 0),
                            'psnr_db': gpu_results.get('psnr_db', 0),
                            'ssim': gpu_results.get('ssim', 0),
                            'compression_ratio': gpu_results.get('compression_ratio', 0),
                            'decode_fps': gpu_results.get('decode_fps', 0),
                            'tops_per_frame': gpu_results.get('tops_per_frame', 0),
                            'gpu_results': gpu_results
                        }
                    
                    elif gpu_status == 'failed':
                        # GPU worker failed
                        error = item.get('gpu_error', 'Unknown GPU error')
                        logger.error(f"   ❌ GPU worker failed: {error}")
                        
                        return {
                            'success': False,
                            'error': error
                        }
                
            except Exception as e:
                logger.warning(f"   ⚠️  Error polling DynamoDB: {e}")
            
            # Wait before next poll
            elapsed = time.time() - start_wait
            if elapsed % 30 == 0:  # Log every 30 seconds
                logger.info(f"   ⏳ Still waiting... ({int(elapsed)}s / {self.gpu_timeout_seconds}s)")
            
            time.sleep(self.poll_interval_seconds)
        
        # Timeout
        logger.error(f"   ❌ GPU execution timeout ({self.gpu_timeout_seconds}s)")
        return {
            'success': False,
            'error': f'GPU execution timeout after {self.gpu_timeout_seconds}s'
        }
    
    def _phase_analysis(
        self,
        experiment_id: str,
        timestamp: int,
        gpu_result: Dict
    ) -> Dict:
        """
        Phase 4: Analyze results and determine next steps.
        
        This phase:
        - Evaluates metrics against targets
        - Calculates improvement over baseline
        - Updates blog post with results
        - Identifies areas for improvement
        """
        try:
            # Extract metrics
            bitrate_mbps = gpu_result.get('bitrate_mbps', 0)
            psnr_db = gpu_result.get('psnr_db', 0)
            ssim = gpu_result.get('ssim', 0)
            compression_ratio = gpu_result.get('compression_ratio', 0)
            tops_per_frame = gpu_result.get('tops_per_frame', 0)
            
            # Targets
            target_bitrate = 1.0  # Mbps
            target_psnr = 35.0    # dB
            target_ssim = 0.95
            target_tops = 1.33    # Per frame at 30 FPS
            
            hevc_baseline = 10.0  # Mbps
            
            # Evaluate
            bitrate_achieved = bitrate_mbps <= target_bitrate
            quality_achieved = psnr_db >= target_psnr and ssim >= target_ssim
            tops_achieved = tops_per_frame <= target_tops
            
            target_achieved = bitrate_achieved and quality_achieved and tops_achieved
            
            # Calculate improvements
            bitrate_reduction_percent = ((hevc_baseline - bitrate_mbps) / hevc_baseline) * 100
            
            logger.info(f"   📈 Metrics:")
            logger.info(f"      Bitrate: {bitrate_mbps:.4f} Mbps (target: ≤{target_bitrate} Mbps) {'✅' if bitrate_achieved else '❌'}")
            logger.info(f"      PSNR: {psnr_db:.2f} dB (target: ≥{target_psnr} dB) {'✅' if psnr_db >= target_psnr else '❌'}")
            logger.info(f"      SSIM: {ssim:.4f} (target: ≥{target_ssim}) {'✅' if ssim >= target_ssim else '❌'}")
            logger.info(f"      TOPS: {tops_per_frame:.4f} (target: ≤{target_tops}) {'✅' if tops_achieved else '❌'}")
            logger.info(f"      Compression: {compression_ratio:.1f}x")
            logger.info(f"      Bitrate reduction: {bitrate_reduction_percent:.1f}% vs HEVC")
            
            # Update blog post with results
            self._write_blog_post_results(
                experiment_id,
                timestamp,
                gpu_result,
                target_achieved
            )
            
            return {
                'success': True,
                'target_achieved': target_achieved,
                'bitrate_mbps': bitrate_mbps,
                'psnr_db': psnr_db,
                'ssim': ssim,
                'compression_ratio': compression_ratio,
                'tops_per_frame': tops_per_frame,
                'bitrate_reduction_percent': bitrate_reduction_percent,
                'gpu_result': gpu_result
            }
            
        except Exception as e:
            logger.error(f"   ❌ Analysis error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
    def _update_experiment_status(
        self,
        experiment_id: str,
        timestamp: int,
        phase: str,
        data: Dict
    ):
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
            
            item = {
                'experiment_id': experiment_id,
                'timestamp': timestamp,
                'current_phase': phase,
                'updated_at': datetime.utcnow().isoformat() + 'Z'
            }
            
            item.update(convert_floats(data))
            
            self.experiments_table.put_item(Item=item)
            
        except Exception as e:
            logger.warning(f"   Failed to update status: {e}")
    
    def _write_blog_post_design(self, experiment_id: str, timestamp: int, llm_analysis: Dict):
        """Write blog post with experiment hypothesis."""
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
            
            experiments_array = [{
                'experiment_type': 'gpu_neural_codec',
                'status': 'running',
                'approach': llm_analysis.get('hypothesis', 'GPU-first two-agent neural codec'),
                'strategy': llm_analysis.get('compression_strategy', 'hybrid'),
                'expected_bitrate': llm_analysis.get('expected_bitrate_mbps', 1.0)
            }]
            
            blog_data = {
                'experiment_id': experiment_id,
                'timestamp': timestamp,
                'timestamp_iso': datetime.utcfromtimestamp(timestamp).isoformat() + 'Z',
                'status': 'running',
                'experiments': json.dumps(experiments_array),
                'phase_completed': 'design'
            }
            
            self.experiments_table.put_item(Item=convert_floats(blog_data))
            logger.info(f"   📝 Blog post created")
            
        except Exception as e:
            logger.warning(f"   Failed to write blog post: {e}")
    
    def _write_blog_post_results(
        self,
        experiment_id: str,
        timestamp: int,
        gpu_result: Dict,
        target_achieved: bool
    ):
        """Update blog post with experiment results."""
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
            
            # Fetch existing blog post to preserve approach
            existing = self.experiments_table.get_item(
                Key={'experiment_id': experiment_id, 'timestamp': timestamp}
            ).get('Item', {})
            
            approach = 'GPU-first two-agent neural codec'
            if existing and 'experiments' in existing:
                existing_exp = json.loads(existing['experiments'])
                if existing_exp:
                    approach = existing_exp[0].get('approach', approach)
            
            experiments_array = [{
                'experiment_type': 'gpu_neural_codec',
                'status': 'completed',
                'approach': approach,
                'metrics': {
                    'bitrate_mbps': gpu_result.get('bitrate_mbps', 0),
                    'psnr_db': gpu_result.get('psnr_db', 0),
                    'ssim': gpu_result.get('ssim', 0),
                    'compression_ratio': gpu_result.get('compression_ratio', 0),
                    'tops_per_frame': gpu_result.get('tops_per_frame', 0)
                },
                'target_achieved': target_achieved
            }]
            
            blog_data = {
                'experiment_id': experiment_id,
                'timestamp': timestamp,
                'timestamp_iso': datetime.utcfromtimestamp(timestamp).isoformat() + 'Z',
                'status': 'completed',
                'experiments': json.dumps(experiments_array),
                'phase_completed': 'analysis',
                'target_achieved': target_achieved
            }
            
            self.experiments_table.put_item(Item=convert_floats(blog_data))
            logger.info(f"   📝 Blog post updated with results")
            
        except Exception as e:
            logger.warning(f"   Failed to update blog post: {e}")
    
    def _create_success_result(self, experiment_id: str, analysis_result: Dict) -> Dict:
        """Create successful experiment result."""
        return {
            'success': True,
            'experiment_id': experiment_id,
            'phase': self.current_phase.value,
            'target_achieved': analysis_result.get('target_achieved', False),
            'metrics': {
                'bitrate_mbps': analysis_result.get('bitrate_mbps', 0),
                'psnr_db': analysis_result.get('psnr_db', 0),
                'ssim': analysis_result.get('ssim', 0),
                'tops_per_frame': analysis_result.get('tops_per_frame', 0)
            }
        }
    
    def _create_failure_result(self, experiment_id: str, reason: str, details: Dict) -> Dict:
        """Create failed experiment result."""
        return {
            'success': False,
            'experiment_id': experiment_id,
            'phase': self.current_phase.value,
            'failure_reason': reason,
            'details': details
        }


def main():
    """Main entry point for GPU-first orchestrator."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    orchestrator = GPUFirstOrchestrator()
    
    iteration = int(os.environ.get('EXPERIMENT_ITERATION', '1'))
    
    logger.info("=" * 80)
    logger.info("🚀 GPU-FIRST ORCHESTRATOR")
    logger.info(f"   Iteration: {iteration}")
    logger.info(f"   No local execution - all work dispatched to GPU workers")
    logger.info("=" * 80)
    
    result = orchestrator.run_experiment_cycle(iteration)
    
    if result['success']:
        logger.info("\n✅ CYCLE COMPLETED")
        if result.get('target_achieved'):
            logger.info("🎯 TARGET ACHIEVED!")
        sys.exit(0)
    else:
        logger.error(f"\n❌ CYCLE FAILED: {result['failure_reason']}")
        sys.exit(1)


if __name__ == '__main__':
    main()


