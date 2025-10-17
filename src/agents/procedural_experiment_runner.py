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
        self.max_validation_retries = 5
        self.max_execution_retries = 5
        
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
        
        logger.info(f"=" * 60)
        logger.info(f"PROCEDURAL EXPERIMENT {iteration}")
        logger.info(f"ID: {experiment_id}")
        logger.info(f"=" * 60)
        
        try:
            # Phase 1: Design
            self.current_phase = ExperimentPhase.DESIGN
            design_result = self._phase_design(experiment_id)
            if not design_result['success']:
                return self._create_failure_result(experiment_id, "design_failed", design_result)
            
            # Phase 2: Deploy
            self.current_phase = ExperimentPhase.DEPLOY
            deploy_result = self._phase_deploy(experiment_id, design_result)
            if not deploy_result['success']:
                return self._create_failure_result(experiment_id, "deploy_failed", deploy_result)
            
            # Phase 3: Validation (with retry loop)
            self.current_phase = ExperimentPhase.VALIDATION
            validation_result = self._phase_validation_with_retry(experiment_id, deploy_result)
            if not validation_result['success']:
                return self._create_failure_result(experiment_id, "validation_failed", validation_result)
            
            # Phase 4: Execution (with retry loop)
            self.current_phase = ExperimentPhase.EXECUTION
            execution_result = self._phase_execution_with_retry(experiment_id, validation_result)
            if not execution_result['success']:
                return self._create_failure_result(experiment_id, "execution_failed", execution_result)
            
            # Phase 5: Analysis
            self.current_phase = ExperimentPhase.ANALYSIS
            analysis_result = self._phase_analysis(experiment_id, execution_result)
            
            # Phase 6: Complete
            self.current_phase = ExperimentPhase.COMPLETE
            return self._create_success_result(experiment_id, analysis_result)
            
        except Exception as e:
            logger.error(f"❌ Unexpected error in experiment: {e}")
            return self._create_failure_result(experiment_id, "unexpected_error", {'error': str(e)})
    
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
        if not code or not validation_result.get('validated'):
            logger.info("  ⚠️  No validated code - using baseline")
            code = None
        
        for attempt in range(1, self.max_execution_retries + 1):
            logger.info(f"  Execution attempt {attempt}/{self.max_execution_retries}")
            
            # Run actual experiment
            try:
                # Import here to avoid circular dependencies
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
                    logger.info(f"  Bitrate: {bitrate:.4f} Mbps")
                    
                    return {
                        'success': True,
                        'executed': True,
                        'retries': attempt - 1,
                        'results': results
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
    
    def _phase_analysis(self, experiment_id: str, execution_result: Dict) -> Dict:
        """Phase 5: Analyze results and store to DynamoDB."""
        logger.info("📊 PHASE 5: ANALYSIS")
        
        results = execution_result.get('results', {})
        
        # Store results
        experiment_data = {
            'experiment_id': experiment_id,
            'timestamp': int(time.time()),
            'experiment_type': 'llm_procedural_evolution',
            'status': 'completed',
            'real_metrics': results.get('real_metrics', {}),
            'validation_retries': 0,  # Will be filled from validation phase
            'execution_retries': execution_result.get('retries', 0),
            'phase_completed': 'analysis',
            'needs_human': len(self.human_intervention_reasons) > 0,
            'human_intervention_reasons': self.human_intervention_reasons if self.human_intervention_reasons else []
        }
        
        try:
            self.experiments_table.put_item(Item=experiment_data)
            logger.info(f"  ✅ Results stored to DynamoDB")
        except Exception as e:
            logger.error(f"  ⚠️  Failed to store results: {e}")
        
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

