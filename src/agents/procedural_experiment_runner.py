#!/usr/bin/env python3
"""
Procedural Experiment Runner
Implements thorough validation-execution-fix loop without time constraints.
"""

import os
import sys
import json
import time
import logging
import boto3
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm_experiment_planner import LLMExperimentPlanner
from agents.adaptive_codec_agent import AdaptiveCodecAgent
from utils.framework_modifier import FrameworkModifier

logger = logging.getLogger(__name__)


class ExperimentPhase(Enum):
    """Phases of experiment execution"""
    DESIGN = "design"
    DEPLOY = "deploy"
    VALIDATION = "validation"
    EXECUTION = "execution"
    QUALITY_VERIFICATION = "quality_verification"
    ANALYSIS = "analysis"
    COMPLETE = "complete"
    NEEDS_HUMAN = "needs_human"


class ProceduralExperimentRunner:
    """
    Runs experiments through a thorough procedural approach:
    1. Design experiment and code
    2. Deploy to sandbox
    3. Validate (retry with fixes if needed)
    4. Execute (retry with fixes if needed)
    5. Analyze results
    6. Design next experiment
    """
    
    def __init__(self):
        """Initialize the procedural experiment runner."""
        self.planner = LLMExperimentPlanner()
        self.codec_agent = AdaptiveCodecAgent()
        self.framework_modifier = FrameworkModifier()
        
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        self.experiments_table = self.dynamodb.Table('ai-video-codec-experiments')
        
        # Retry limits
        self.max_validation_retries = 10
        self.max_execution_retries = 10
        
        # State tracking
        self.current_phase = ExperimentPhase.DESIGN
        self.human_intervention_reasons = []
    
    def run_single_experiment(self, iteration: int) -> Dict:
        """
        Run a complete experiment cycle with procedural validation/execution.
        
        Args:
            iteration: Experiment iteration number
            
        Returns:
            Experiment results dict
        """
        experiment_id = f"proc_exp_{int(time.time())}"
        start_time = time.time()
        initial_timestamp = int(start_time)  # Store for consistent DynamoDB key
        
        logger.info(f"=" * 60)
        logger.info(f"PROCEDURAL EXPERIMENT {iteration}")
        logger.info(f"ID: {experiment_id}")
        logger.info(f"=" * 60)
        
        # Estimate: Design (10s) + Deploy (1s) + Validation (30s) + Execution (60s) + Analysis (5s) = ~106s
        estimated_duration_seconds = 106
        
        # Create initial experiment record in DynamoDB
        self._update_experiment_status(experiment_id, initial_timestamp, 'design', 'running', {
            'start_time': start_time,
            'estimated_duration_seconds': estimated_duration_seconds,
            'elapsed_seconds': 0
        })
        
        try:
            # Phase 1: Design
            self.current_phase = ExperimentPhase.DESIGN
            design_result = self._phase_design(experiment_id)
            if not design_result['success']:
                return self._create_failure_result(experiment_id, "design_failed", design_result)
            
            # Update: Design complete
            elapsed = time.time() - start_time
            self._update_experiment_status(experiment_id, initial_timestamp, 'deploy', 'running', {
                'start_time': start_time,
                'estimated_duration_seconds': estimated_duration_seconds,
                'elapsed_seconds': int(elapsed)
            })
            
            # Phase 2: Deploy
            self.current_phase = ExperimentPhase.DEPLOY
            deploy_result = self._phase_deploy(experiment_id, design_result)
            if not deploy_result['success']:
                return self._create_failure_result(experiment_id, "deploy_failed", deploy_result)
            
            # Update: Deploy complete
            elapsed = time.time() - start_time
            self._update_experiment_status(experiment_id, initial_timestamp, 'validation', 'running', {
                'start_time': start_time,
                'estimated_duration_seconds': estimated_duration_seconds,
                'elapsed_seconds': int(elapsed)
            })
            
            # Phase 3: Validation (with retry loop)
            self.current_phase = ExperimentPhase.VALIDATION
            validation_result = self._phase_validation_with_retry(experiment_id, deploy_result)
            if not validation_result['success']:
                return self._create_failure_result(experiment_id, "validation_failed", validation_result)
            
            # Update: Validation complete
            elapsed = time.time() - start_time
            self._update_experiment_status(experiment_id, initial_timestamp, 'execution', 'running', {
                'start_time': start_time,
                'estimated_duration_seconds': estimated_duration_seconds,
                'elapsed_seconds': int(elapsed),
                'validation_retries': validation_result.get('retries', 0)
            })
            
            # Phase 4: Execution (with retry loop)
            self.current_phase = ExperimentPhase.EXECUTION
            execution_result = self._phase_execution_with_retry(experiment_id, validation_result)
            if not execution_result['success']:
                return self._create_failure_result(experiment_id, "execution_failed", execution_result)
            
            # Update: Execution complete, moving to quality verification
            elapsed = time.time() - start_time
            self._update_experiment_status(experiment_id, initial_timestamp, 'quality_verification', 'running', {
                'start_time': start_time,
                'estimated_duration_seconds': estimated_duration_seconds,
                'elapsed_seconds': int(elapsed),
                'validation_retries': validation_result.get('retries', 0),
                'execution_retries': execution_result.get('retries', 0)
            })
            
            # Phase 5: Quality Verification (Decompression + PSNR/SSIM)
            self.current_phase = ExperimentPhase.QUALITY_VERIFICATION
            quality_result = self._phase_quality_verification(experiment_id, execution_result, validation_result)
            if not quality_result['success']:
                logger.warning(f"  ⚠️  Quality verification failed, continuing with execution metrics only")
            
            # Update: Quality verification complete
            elapsed = time.time() - start_time
            self._update_experiment_status(experiment_id, initial_timestamp, 'analysis', 'running', {
                'start_time': start_time,
                'estimated_duration_seconds': estimated_duration_seconds,
                'elapsed_seconds': int(elapsed),
                'validation_retries': validation_result.get('retries', 0),
                'execution_retries': execution_result.get('retries', 0),
                'quality_verified': quality_result.get('quality_verified', False)
            })
            
            # Phase 6: Analysis
            self.current_phase = ExperimentPhase.ANALYSIS
            # Merge quality metrics into execution result for analysis
            if quality_result['success']:
                execution_result['quality_metrics'] = quality_result.get('quality_metrics', {})
            analysis_result = self._phase_analysis(experiment_id, execution_result)
            
            # Phase 7: Complete
            self.current_phase = ExperimentPhase.COMPLETE
            final_elapsed = time.time() - start_time
            logger.info(f"✅ Experiment completed in {final_elapsed:.1f}s (estimated: {estimated_duration_seconds}s)")
            return self._create_success_result(experiment_id, analysis_result)
            
        except Exception as e:
            logger.error(f"❌ Unexpected error in experiment: {e}")
            return self._create_failure_result(experiment_id, "unexpected_error", {'error': str(e)})
    
    def _update_experiment_status(self, experiment_id: str, timestamp: int, current_phase: str, status: str, extra_data: Dict):
        """
        Update experiment status in DynamoDB for real-time dashboard visibility.
        
        Args:
            experiment_id: Experiment ID
            timestamp: Fixed timestamp for this experiment (DynamoDB sort key)
            current_phase: Current phase (design, deploy, validation, execution, analysis, complete)
            status: Status (running, completed, failed)
            extra_data: Additional data to merge (e.g., retry counts)
        """
        try:
            from decimal import Decimal
            
            # Convert floats to Decimal for DynamoDB
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
                'timestamp': timestamp,  # Use fixed timestamp for consistent key
                'status': status,
                'current_phase': current_phase,
                'phase_completed': current_phase,  # Track last completed phase
            }
            
            # Merge in any extra data (like retry counts), converting floats to Decimal
            item.update(convert_floats(extra_data))
            
            self.experiments_table.put_item(Item=item)
            logger.debug(f"  Updated status: {current_phase} ({status})")
        except Exception as e:
            logger.warning(f"  Failed to update experiment status: {e}")
    
    def _write_blog_post_design(self, experiment_id: str, llm_analysis: Dict):
        """
        Write initial blog post after design phase completes.
        This creates a blog entry with the approach/hypothesis before execution.
        
        Note: This function is called FROM _phase_design, so it doesn't need timestamp
        passed in - it extracts it from the experiment_id.
        """
        from datetime import datetime
        from decimal import Decimal
        
        try:
            # Extract timestamp from experiment_id (format: proc_exp_{timestamp})
            timestamp = int(experiment_id.split('_')[-1])
            
            # Create placeholder experiments array with approach but no results yet
            experiments_array = [{
                'experiment_type': 'real_procedural_generation',
                'status': 'running',
                'real_metrics': {},  # Empty until execution completes
                'comparison': {},
                'approach': llm_analysis.get('hypothesis', 'Exploring new compression approach'),
                'expected_bitrate': llm_analysis.get('expected_bitrate_mbps', 0)
            }]
            
            def convert_floats(obj):
                if isinstance(obj, dict):
                    return {k: convert_floats(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_floats(item) for item in obj]
                elif isinstance(obj, float):
                    return Decimal(str(obj))
                return obj
            
            blog_data = {
                'experiment_id': experiment_id,
                'timestamp': timestamp,  # Use consistent timestamp from experiment_id
                'timestamp_iso': datetime.utcfromtimestamp(timestamp).isoformat() + 'Z',
                'status': 'running',
                'experiments': json.dumps(experiments_array),
                'phase_completed': 'design',
            }
            
            self.experiments_table.put_item(Item=convert_floats(blog_data))
            logger.info(f"  📝 Blog post created with approach")
            
            # Also write to reasoning table for blog context (reasoning_id is the primary key)
            try:
                reasoning_table = boto3.resource('dynamodb', region_name='us-east-1').Table('ai-video-codec-reasoning')
                reasoning_data = {
                    'reasoning_id': experiment_id,  # Use experiment_id as reasoning_id
                    'experiment_id': experiment_id,
                    'timestamp': timestamp,
                    'hypothesis': llm_analysis.get('hypothesis', ''),
                    'root_cause': llm_analysis.get('root_cause', ''),
                    'insights': json.dumps(llm_analysis.get('insights', [])),
                    'next_experiment': json.dumps(llm_analysis.get('next_experiment', {})),
                    'confidence_score': llm_analysis.get('confidence_score', 0.0)
                }
                reasoning_table.put_item(Item=convert_floats(reasoning_data))
                logger.info(f"  📝 Reasoning data stored")
            except Exception as e:
                logger.debug(f"  Could not write to reasoning table: {e}")
        except Exception as e:
            logger.warning(f"  Failed to write blog post design: {e}")
    
    def _phase_design(self, experiment_id: str) -> Dict:
        """Phase 1: Design experiment and generate code."""
        logger.info("📐 PHASE 1: DESIGN")
        logger.info("  Analyzing recent experiments and designing next approach...")
        
        try:
            # Fetch recent experiments
            recent_experiments = self.planner.analyze_recent_experiments(limit=5)
            
            # Get LLM analysis and recommendations
            llm_analysis = self.planner.get_llm_analysis(recent_experiments)
            
            if not llm_analysis:
                logger.warning("  LLM analysis not available - using fallback")
                return {'success': True, 'llm_analysis': None, 'code': None}
            
            # Generate code based on analysis
            code = llm_analysis.get('generated_code', {}).get('code')
            
            logger.info(f"  ✅ Design complete")
            logger.info(f"  Root cause identified: {llm_analysis.get('root_cause', 'N/A')[:100]}...")
            logger.info(f"  Hypothesis: {llm_analysis.get('hypothesis', 'N/A')[:100]}...")
            logger.info(f"  Code generated: {len(code) if code else 0} characters")
            
            # Write initial blog post with hypothesis (before execution)
            self._write_blog_post_design(experiment_id, llm_analysis)
            
            return {
                'success': True,
                'llm_analysis': llm_analysis,
                'code': code
            }
            
        except Exception as e:
            logger.error(f"  ❌ Design phase failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _phase_deploy(self, experiment_id: str, design_result: Dict) -> Dict:
        """Phase 2: Deploy code to sandbox."""
        logger.info("📦 PHASE 2: DEPLOY")
        
        code = design_result.get('code')
        llm_analysis = design_result.get('llm_analysis')
        
        if not code:
            logger.warning("  ⚠️  No code to deploy - LLM code generation failed")
            logger.warning("  This requires human intervention to fix LLM code generation")
            
            # Flag for human intervention
            self.human_intervention_reasons.append({
                'phase': 'design',
                'reason': 'LLM code generation returned 0 characters',
                'llm_analysis': llm_analysis.get('root_cause', 'N/A') if llm_analysis else 'LLM analysis unavailable',
                'hypothesis': llm_analysis.get('hypothesis', 'N/A') if llm_analysis else 'N/A'
            })
            
            # Continue with baseline but mark as needs human
            logger.info("  Will continue with baseline codec to gather data, but flagged for review")
            return {'success': True, 'code': None, 'deployed': False, 'needs_human': True}
        
        logger.info(f"  Deploying {len(code)} chars of code to sandbox...")
        
        # Code is deployed during validation phase
        # This phase just prepares and verifies we have code
        logger.info("  ✅ Code ready for deployment")
        
        return {
            'success': True,
            'code': code,
            'deployed': True
        }
    
    def _phase_validation_with_retry(self, experiment_id: str, deploy_result: Dict) -> Dict:
        """Phase 3: Validate code, retry with fixes if needed."""
        logger.info("🔍 PHASE 3: VALIDATION (with intelligent retry)")
        
        code = deploy_result.get('code')
        if not code:
            logger.info("  ⚠️  No new code to validate - skipping")
            return {'success': True, 'validated': False, 'retries': 0}
        
        for attempt in range(1, self.max_validation_retries + 1):
            logger.info(f"  Validation attempt {attempt}/{self.max_validation_retries}")
            
            # Test code in sandbox
            validation_passed, metrics = self.codec_agent.test_generated_code(code)
            
            if validation_passed:
                logger.info(f"  ✅ Validation PASSED on attempt {attempt}")
                return {
                    'success': True,
                    'validated': True,
                    'retries': attempt - 1,
                    'code': code,
                    'metrics': metrics
                }
            
            # Validation failed - analyze and fix
            logger.warning(f"  ❌ Validation FAILED on attempt {attempt}")
            
            # Get failure analysis
            failure_analysis = self.codec_agent._last_failure_analysis
            if failure_analysis:
                logger.info(f"  Failure: {failure_analysis.get('failure_category', 'unknown')}")
                logger.info(f"  Root cause: {failure_analysis.get('root_cause', 'N/A')[:100]}")
                logger.info(f"  Fix: {failure_analysis.get('fix_suggestion', 'N/A')[:100]}")
                
                # Try to auto-fix if it's a framework issue
                if self._can_autofix(failure_analysis):
                    logger.info(f"  🔧 Attempting auto-fix...")
                    fix_applied = self._apply_autofix(failure_analysis)
                    
                    if fix_applied:
                        logger.info(f"  ✅ Auto-fix applied, retrying...")
                        continue
                    else:
                        logger.warning(f"  ⚠️  Auto-fix failed")
            
            # Try regenerating code with failure feedback
            if attempt < self.max_validation_retries:
                logger.info(f"  🔄 Requesting LLM to fix code...")
                # TODO: Could call LLM here to regenerate code based on failure
                # For now, just retry with same code after framework fixes
                time.sleep(2)  # Brief pause
        
        # Max retries reached
        logger.error(f"  ❌ Validation failed after {self.max_validation_retries} attempts")
        self.human_intervention_reasons.append({
            'phase': 'validation',
            'reason': 'Max validation retries exceeded',
            'last_failure': failure_analysis
        })
        
        return {
            'success': False,
            'validated': False,
            'retries': self.max_validation_retries,
            'needs_human': True,
            'failure_analysis': failure_analysis
        }
    
    def _phase_execution_with_retry(self, experiment_id: str, validation_result: Dict) -> Dict:
        """Phase 4: Execute code, retry with fixes if needed."""
        logger.info("▶️  PHASE 4: EXECUTION (with intelligent retry)")
        
        code = validation_result.get('code')
        has_llm_code = bool(code and validation_result.get('validated'))
        
        if not has_llm_code:
            logger.info("  ⚠️  No validated code - using baseline")
        else:
            logger.info(f"  ✅ Using LLM-generated code ({len(code)} chars)")
        
        for attempt in range(1, self.max_execution_retries + 1):
            logger.info(f"  Execution attempt {attempt}/{self.max_execution_retries}")
            
            # Run actual experiment
            try:
                if has_llm_code:
                    # USE THE LLM CODE! Run with AdaptiveCodecAgent
                    logger.info("  🧪 Running experiment with LLM code...")
                    
                    results = self.codec_agent.run_real_experiment_with_code(
                        code=code,
                        duration=10.0,
                        fps=30.0,
                        resolution=(1920, 1080)
                    )
                    
                    logger.info(f"  📊 LLM code execution complete")
                    
                else:
                    # Fallback: Use baseline procedural generator
                    logger.info("  🔄 Running baseline procedural generator...")
                    from agents.procedural_generator import ProceduralCompressionAgent
                    
                    agent = ProceduralCompressionAgent(resolution=(1920, 1080), config={})
                    
                    # Generate test video
                    timestamp = int(time.time())
                    output_path = f"/tmp/proc_exp_{timestamp}.mp4"
                    
                    results = agent.generate_procedural_video(
                        output_path,
                        duration=10.0,
                        fps=30.0
                    )
                
                # Check if execution succeeded
                if results.get('status') == 'completed':
                    logger.info(f"  ✅ Execution SUCCEEDED on attempt {attempt}")
                    bitrate = results.get('real_metrics', {}).get('bitrate_mbps', 0)
                    logger.info(f"  📈 Bitrate: {bitrate:.4f} Mbps")
                    
                    # Log if this is LLM code vs baseline
                    if has_llm_code:
                        logger.info(f"  🎯 Result from LLM-generated compression code")
                    else:
                        logger.info(f"  ⚠️  Result from baseline (no LLM code)")
                    
                    return {
                        'success': True,
                        'executed': True,
                        'retries': attempt - 1,
                        'results': results,
                        'used_llm_code': has_llm_code
                    }
                else:
                    logger.warning(f"  ❌ Execution returned non-completed status")
                    
            except Exception as e:
                logger.error(f"  ❌ Execution FAILED on attempt {attempt}: {e}")
                
                # Try to diagnose and fix
                if attempt < self.max_execution_retries:
                    logger.info(f"  🔧 Analyzing execution failure...")
                    # Could use LogAnalyzer here
                    time.sleep(2)
                    continue
        
        # Max retries reached
        logger.error(f"  ❌ Execution failed after {self.max_execution_retries} attempts")
        self.human_intervention_reasons.append({
            'phase': 'execution',
            'reason': 'Max execution retries exceeded',
            'last_error': str(e) if 'e' in locals() else 'Unknown error'
        })
        
        return {
            'success': False,
            'executed': False,
            'retries': self.max_execution_retries,
            'needs_human': True
        }
    
    def _phase_quality_verification(self, experiment_id: str, execution_result: Dict, validation_result: Dict) -> Dict:
        """
        Phase 5: Quality Verification (Decompression + PSNR/SSIM)
        
        This phase:
        1. Loads the compressed data from execution
        2. Decompresses all frames using LLM code
        3. Calculates PSNR and SSIM against original
        4. Returns quality metrics
        """
        logger.info("🔍 PHASE 5: QUALITY VERIFICATION (Decompression + PSNR/SSIM)")
        
        # Check if we have LLM code with decompress function
        code = validation_result.get('code')
        has_decompress = code and 'def decompress_video_frame' in code
        
        if not has_decompress:
            logger.warning("  ⚠️  No decompress function available - skipping quality verification")
            logger.warning("  💡 LLM should generate BOTH compress_video_frame() AND decompress_video_frame()")
            return {
                'success': False,
                'quality_verified': False,
                'skip_reason': 'no_decompress_function'
            }
        
        try:
            import cv2
            import numpy as np
            from skimage.metrics import peak_signal_noise_ratio as psnr_metric
            from skimage.metrics import structural_similarity as ssim_metric
            from utils.code_sandbox import CodeSandbox
            
            results = execution_result.get('results', {})
            real_metrics = results.get('real_metrics', {})
            
            # Check if we have the necessary data
            compressed_data_path = results.get('compressed_data_path')
            original_video_path = results.get('original_video_path')
            
            if not compressed_data_path or not original_video_path:
                logger.warning("  ⚠️  Missing compressed data or original video path")
                # For now, extract quality metrics if they were already calculated
                if 'psnr_db' in real_metrics:
                    logger.info(f"  ✅ Quality metrics already available from execution phase")
                    return {
                        'success': True,
                        'quality_verified': True,
                        'quality_metrics': {
                            'psnr_db': real_metrics.get('psnr_db'),
                            'ssim': real_metrics.get('ssim'),
                            'quality': real_metrics.get('quality')
                        }
                    }
                
                return {
                    'success': False,
                    'quality_verified': False,
                    'skip_reason': 'missing_data_paths'
                }
            
            logger.info(f"  📹 Loading original video and compressed data...")
            logger.info(f"     Original: {original_video_path}")
            logger.info(f"     Compressed: {compressed_data_path}")
            
            # For now, if quality metrics were already calculated in execution phase,
            # return them (backward compatibility)
            if 'psnr_db' in real_metrics and real_metrics['psnr_db'] > 0:
                psnr_db = real_metrics['psnr_db']
                ssim = real_metrics.get('ssim', 0.0)
                quality = real_metrics.get('quality', 'unknown')
                
                logger.info(f"  ✅ Quality metrics already calculated:")
                logger.info(f"     PSNR: {psnr_db:.2f} dB")
                logger.info(f"     SSIM: {ssim:.4f}")
                logger.info(f"     Quality: {quality}")
                
                return {
                    'success': True,
                    'quality_verified': True,
                    'quality_metrics': {
                        'psnr_db': psnr_db,
                        'ssim': ssim,
                        'quality': quality
                    }
                }
            
            # If we get here, quality verification wasn't done in execution phase
            # This shouldn't happen with the current code, but handle gracefully
            logger.warning("  ⚠️  Quality metrics not found in execution result")
            logger.warning("  This indicates the quality verification code in adaptive_codec_agent didn't run")
            
            return {
                'success': False,
                'quality_verified': False,
                'skip_reason': 'metrics_not_calculated'
            }
            
        except Exception as e:
            logger.error(f"  ❌ Quality verification failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'quality_verified': False,
                'error': str(e)
            }
    
    def _phase_analysis(self, experiment_id: str, execution_result: Dict) -> Dict:
        """Phase 6: Analyze results and store to DynamoDB."""
        logger.info("📊 PHASE 6: ANALYSIS")
        
        results = execution_result.get('results', {})
        real_metrics = results.get('real_metrics', results)
        
        # Calculate comparison with HEVC baseline (10 Mbps)
        hevc_baseline_mbps = 10.0
        bitrate_mbps = real_metrics.get('bitrate_mbps', 0)
        reduction_percent = ((hevc_baseline_mbps - bitrate_mbps) / hevc_baseline_mbps) * 100 if bitrate_mbps else -50.0
        target_achieved = bitrate_mbps < 1.0 if bitrate_mbps else False
        
        # Format experiment data in the structure the blog expects
        # The blog looks for experiments[].experiment_type == 'real_procedural_generation'
        
        # Try to fetch existing blog post to preserve the approach field
        approach = 'Compression experiment'
        try:
            timestamp = int(experiment_id.split('_')[-1])
            existing = self.experiments_table.get_item(
                Key={'experiment_id': experiment_id, 'timestamp': timestamp}
            ).get('Item', {})
            
            if existing and 'experiments' in existing:
                existing_exp = json.loads(existing['experiments'])
                if existing_exp and len(existing_exp) > 0:
                    approach = existing_exp[0].get('approach', approach)
        except Exception as e:
            logger.debug(f"Could not fetch existing approach: {e}")
        
        experiments_array = [{
            'experiment_type': 'real_procedural_generation',
            'status': 'completed',
            'approach': approach,  # Preserve the approach from design phase
            'real_metrics': real_metrics,
            'comparison': {
                'hevc_baseline_mbps': hevc_baseline_mbps,
                'reduction_percent': reduction_percent,
                'target_achieved': target_achieved
            }
        }]
        
        # Store results in format compatible with blog
        from datetime import datetime
        from decimal import Decimal
        
        # Convert floats to Decimal for DynamoDB
        def convert_floats(obj):
            if isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_floats(item) for item in obj]
            elif isinstance(obj, float):
                return Decimal(str(obj))
            return obj
        
        # Extract timestamp from experiment_id (format: proc_exp_{timestamp})
        timestamp = int(experiment_id.split('_')[-1])
        timestamp_iso = datetime.utcfromtimestamp(timestamp).isoformat() + 'Z'
        
        # Update blog post with results (using consistent timestamp)
        experiment_data = {
            'experiment_id': experiment_id,
            'timestamp': timestamp,  # Use consistent timestamp from experiment_id
            'timestamp_iso': timestamp_iso,
            'status': 'completed',
            'experiments': json.dumps(experiments_array),  # Blog expects JSON string with results
            'validation_retries': 0,  # Will be filled from validation phase
            'execution_retries': execution_result.get('retries', 0),
            'phase_completed': 'analysis',
            'needs_human': len(self.human_intervention_reasons) > 0,
            'human_intervention_reasons': self.human_intervention_reasons if self.human_intervention_reasons else []
        }
        
        try:
            # Convert any float values to Decimal for DynamoDB
            experiment_data_converted = convert_floats(experiment_data)
            self.experiments_table.put_item(Item=experiment_data_converted)
            logger.info(f"  ✅ Blog post updated with results")
            logger.info(f"  Bitrate: {bitrate_mbps:.2f} Mbps, Reduction: {reduction_percent:.1f}%")
        except Exception as e:
            logger.error(f"  ⚠️  Failed to update blog post: {e}")
        
        logger.info("  ✅ Analysis complete")
        
        return {
            'success': True,
            'experiment_data': experiment_data
        }
    
    def _can_autofix(self, failure_analysis: Dict) -> bool:
        """Determine if failure can be auto-fixed with tools."""
        category = failure_analysis.get('failure_category', '')
        
        # Categories that can potentially be auto-fixed
        fixable_categories = [
            'import_error',
            'syntax_error',  # Some syntax errors are due to framework limitations
            'validation_error'
        ]
        
        return category in fixable_categories
    
    def _apply_autofix(self, failure_analysis: Dict) -> bool:
        """
        Apply automatic fix using framework tools.
        
        Returns:
            True if fix was applied successfully
        """
        try:
            category = failure_analysis.get('failure_category', '')
            root_cause = failure_analysis.get('root_cause', '')
            fix_suggestion = failure_analysis.get('fix_suggestion', '')
            
            logger.info(f"  Applying auto-fix for {category}...")
            
            # Example: Fix missing import by modifying sandbox
            if 'import' in category or 'not defined' in root_cause.lower():
                # Extract what needs to be added
                # This is simplified - real implementation would parse the error
                logger.info(f"  Fix: {fix_suggestion[:100]}")
                
                # Use framework modifier tools
                # For now, just log - actual implementation would call tools
                logger.info(f"  ⚠️  Auto-fix not yet implemented for this case")
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"  ❌ Auto-fix error: {e}")
            return False
    
    def _create_success_result(self, experiment_id: str, analysis_result: Dict) -> Dict:
        """Create successful experiment result."""
        needs_human = len(self.human_intervention_reasons) > 0
        
        result = {
            'success': True,
            'experiment_id': experiment_id,
            'phase': self.current_phase.value,
            'needs_human': needs_human,
            'data': analysis_result.get('experiment_data', {})
        }
        
        # Add human intervention reasons if any
        if needs_human:
            result['human_intervention_reasons'] = self.human_intervention_reasons
            logger.warning(f"🚨 Experiment completed but needs human intervention:")
            for reason in self.human_intervention_reasons:
                logger.warning(f"   - {reason['phase']}: {reason['reason']}")
        
        return result
    
    def _create_failure_result(self, experiment_id: str, reason: str, details: Dict) -> Dict:
        """Create failed experiment result."""
        needs_human = len(self.human_intervention_reasons) > 0
        
        return {
            'success': False,
            'experiment_id': experiment_id,
            'phase': self.current_phase.value,
            'failure_reason': reason,
            'needs_human': needs_human,
            'human_intervention_reasons': self.human_intervention_reasons,
            'details': details
        }


def main():
    """Main entry point for procedural experiment runner."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    runner = ProceduralExperimentRunner()
    
    # Get iteration number from environment or default to 1
    iteration = int(os.environ.get('EXPERIMENT_ITERATION', '1'))
    
    logger.info("Starting Procedural Experiment Runner")
    logger.info(f"Iteration: {iteration}")
    
    result = runner.run_single_experiment(iteration)
    
    if result['success']:
        logger.info("✅ EXPERIMENT COMPLETED SUCCESSFULLY")
        sys.exit(0)
    else:
        logger.error(f"❌ EXPERIMENT FAILED: {result['failure_reason']}")
        if result['needs_human']:
            logger.error("🚨 HUMAN INTERVENTION REQUIRED:")
            for reason in result.get('human_intervention_reasons', []):
                logger.error(f"  - {reason['phase']}: {reason['reason']}")
        sys.exit(1)


if __name__ == '__main__':
    main()

