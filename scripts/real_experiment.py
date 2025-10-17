#!/usr/bin/env python3
"""
Real AI Codec Experiment
Actually runs the neural networks and procedural generation to get real results.
"""

import os
import sys
import json
import time
import logging
import boto3
import numpy as np
from datetime import datetime
from typing import Optional, Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_real_procedural_experiment(llm_config: Optional[Dict] = None):
    """Run actual procedural generation experiment with LLM-suggested configuration."""
    logger.info("Starting REAL procedural generation experiment...")
    
    try:
        from agents.procedural_generator import ProceduralCompressionAgent
        
        # Extract configuration from LLM suggestions
        config = {}
        if llm_config:
            # Parse LLM suggestions into actionable parameters
            next_exp = llm_config.get('next_experiment', {})
            approach = next_exp.get('approach', '').lower()
            hypothesis = llm_config.get('hypothesis', '').lower()
            changes = str(next_exp.get('changes', [])).lower()
            
            # Map LLM suggestions to agent configuration
            # Check if LLM suggests parameter/compact storage approach
            if any(keyword in approach + hypothesis + changes for keyword in ['parameter', 'compact', 'procedural command', 'generation parameter']):
                config['parameter_storage'] = True
                config['compression_strategy'] = 'parameter_storage'
                logger.info("🔧 Enabling parameter storage based on LLM suggestion")
            
            config['bitrate_target_mbps'] = llm_config.get('expected_bitrate_mbps', 1.0)
            config['complexity_level'] = llm_config.get('complexity_level', 1.0)
            
            logger.info(f"📝 LLM Configuration applied: {config}")
        
        # Create procedural agent with LLM configuration
        agent = ProceduralCompressionAgent(resolution=(1920, 1080), config=config)
        logger.info("Procedural agent created successfully")
        
        # Generate unique output filename using timestamp
        import time
        timestamp = int(time.time())
        output_path = f"/tmp/procedural_{timestamp}.mp4"
        
        # Generate procedural video
        logger.info(f"Generating procedural video: {output_path}")
        results = agent.generate_procedural_video(
            output_path, 
            duration=10.0, 
            fps=30.0
        )
        
        logger.info(f"Procedural video generated: {results}")
        
        # NEW: Check if parameter storage mode was used
        if results.get('mode') == 'parameter_storage':
            # Parameter storage mode: use metrics from the agent
            bitrate_mbps = results.get('bitrate_mbps', 15.0)
            file_size_kb = results.get('params_size_kb', 0)
            compression_method = 'parameter_storage'
            parameter_file = results.get('parameter_file', f"/tmp/params_{timestamp}.json")
            output_file = parameter_file
            
            logger.info(f"🎯 PARAMETER STORAGE: {parameter_file} ({file_size_kb:.2f} KB, {bitrate_mbps:.4f} Mbps)")
        else:
            # Old mode: rendered video file - use the actual output path
            file_size = os.path.getsize(output_path)
            bitrate_mbps = (file_size * 8) / (10.0 * 1_000_000)
            file_size_kb = file_size / 1024
            compression_method = 'procedural_demoscene'
            output_file = output_path
            
            logger.info(f"📹 RENDERED VIDEO: {output_path} ({file_size_kb:.2f} KB, {bitrate_mbps:.2f} Mbps)")
        
        real_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'experiment_type': 'real_procedural_generation',
            'status': 'completed',
            'config': config,  # Include config that was used
            'real_metrics': {
                'file_size_kb': file_size_kb,
                'file_size_mb': file_size_kb / 1024,
                'bitrate_mbps': bitrate_mbps,
                'duration': 10.0,
                'fps': 30.0,
                'resolution': '1920x1080',
                'compression_method': compression_method,
                'parameter_storage': results.get('mode') == 'parameter_storage',
                'output_path': output_file,
                'unique_file_id': f"{timestamp}_{os.path.basename(output_file)}"
            },
            'comparison': {
                'hevc_baseline_mbps': 10.0,
                'reduction_percent': ((10.0 - bitrate_mbps) / 10.0) * 100,
                'target_achieved': bitrate_mbps < 1.0
            }
        }
        
        logger.info(f"Real experiment results: {real_results}")
        return real_results
        
    except Exception as e:
        logger.error(f"Real experiment failed: {e}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'experiment_type': 'real_procedural_generation',
            'status': 'failed',
            'error': str(e)
        }

def run_llm_generated_experiment(llm_config: Optional[Dict] = None):
    """Run experiment with LLM-generated compression code and auto-adopt if better."""
    logger.info("Starting LLM-GENERATED CODE experiment...")
    
    try:
        from agents.adaptive_codec_agent import AdaptiveCodecAgent
        from agents.llm_self_debugger import LLMSelfDebugger
        import numpy as np
        import cv2
        
        # SELF-GOVERNANCE: Analyze previous failures first
        debugger = LLMSelfDebugger()
        failure_analysis = debugger.analyze_recent_failures(lookback_hours=1)
        
        if failure_analysis['total_failures'] > 5:
            logger.warning(f"⚠️  Detected {failure_analysis['total_failures']} recent failures")
            logger.warning("🔧 Generating self-governance report...")
            
            governance_report = debugger.create_self_governance_report()
            
            # Log recommendations
            for rec in governance_report['failure_analysis'].get('recommendations', []):
                logger.warning(f"   💡 {rec['description']}: {rec['fix']}")
        
        # Check if LLM generated code
        if not llm_config or 'generated_code' not in llm_config:
            logger.warning("No generated code available - skipping")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'experiment_type': 'llm_generated_code',
                'status': 'skipped',
                'reason': 'no_generated_code',
                'debug_info': failure_analysis
            }
        
        # Create adaptive agent
        adaptive_agent = AdaptiveCodecAgent()
        
        # Evaluate and potentially adopt the new code
        code_info = llm_config['generated_code']
        evolution_result = adaptive_agent.evolve_with_llm_code(code_info)
        
        logger.info(f"Evolution result: {evolution_result['status']}")
        
        if evolution_result.get('adopted'):
            logger.info(f"🎉 NEW CODEC ARCHITECTURE ADOPTED! Version {evolution_result['version']}")
            logger.info(f"Improvement: {evolution_result.get('improvement')}")
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'experiment_type': 'llm_generated_code_evolution',
            'status': 'completed',
            'evolution': evolution_result,
            'code_info': {
                'function_name': code_info.get('function_name'),
                'code_length': len(code_info.get('code', '')),
                'version': evolution_result.get('version', 0)
            }
        }
        
    except Exception as e:
        logger.error(f"LLM-generated experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'experiment_type': 'llm_generated_code',
            'status': 'failed',
            'error': str(e)
        }

def run_real_ai_experiment():
    """Run actual AI neural network experiment."""
    logger.info("Starting REAL AI neural network experiment...")
    
    try:
        import torch
        from agents.ai_codec_agent import SemanticEncoder, MotionPredictor, GenerativeRefiner
        
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Create neural networks
        semantic_encoder = SemanticEncoder()
        motion_predictor = MotionPredictor()
        generative_refiner = GenerativeRefiner()
        
        logger.info("Neural networks created successfully")
        
        # Test with dummy data
        dummy_frame = torch.randn(1, 3, 64, 64)
        
        # Test semantic encoder
        with torch.no_grad():
            semantic_features = semantic_encoder(dummy_frame)
            logger.info(f"Semantic encoder output shape: {semantic_features.shape}")
        
        # Test motion predictor
        with torch.no_grad():
            motion_vectors = motion_predictor(dummy_frame, dummy_frame)
            logger.info(f"Motion predictor output shape: {motion_vectors.shape}")
        
        # Test generative refiner
        with torch.no_grad():
            refined_frame = generative_refiner(dummy_frame)
            logger.info(f"Generative refiner output shape: {refined_frame.shape}")
        
        ai_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'experiment_type': 'real_ai_neural_networks',
            'status': 'completed',
            'neural_networks': {
                'semantic_encoder': 'working',
                'motion_predictor': 'working', 
                'generative_refiner': 'working',
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available()
            }
        }
        
        logger.info(f"AI neural networks test completed: {ai_results}")
        return ai_results
        
    except Exception as e:
        logger.error(f"AI experiment failed: {e}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'experiment_type': 'real_ai_neural_networks',
            'status': 'failed',
            'error': str(e)
        }

def upload_real_results(results):
    """Upload real experiment results to S3 and DynamoDB."""
    try:
        s3 = boto3.client('s3')
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        
        # Create results file
        results_file = f"/tmp/real_experiment_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Upload to S3
        bucket = 'ai-video-codec-videos-580473065386'
        key = f"results/real_experiment_{int(time.time())}.json"
        
        s3.upload_file(results_file, bucket, key)
        logger.info(f"Real results uploaded to s3://{bucket}/{key}")
        
        # Extract code evolution info from experiments
        code_evolution_info = None
        for exp in results['experiments']:
            if exp.get('experiment_type') == 'llm_generated_code_evolution' and exp.get('status') == 'completed':
                evolution = exp.get('evolution', {})
                code_evolution_info = {
                    'code_changed': evolution.get('adopted', False),
                    'version': evolution.get('version', 0),
                    'status': evolution.get('status', 'unknown'),
                    'improvement': evolution.get('improvement', 'N/A'),
                    'summary': evolution.get('summary', 'No changes'),
                    'deployment_status': evolution.get('deployment_status', 'not_deployed'),
                    'github_committed': evolution.get('github_committed', False),
                    'github_commit_hash': evolution.get('github_commit_hash', None)
                }
                break
        
        # If no successful evolution, check for attempts/failures
        if not code_evolution_info:
            for exp in results['experiments']:
                if 'llm' in exp.get('experiment_type', '').lower():
                    code_evolution_info = {
                        'code_changed': False,
                        'version': 0,
                        'status': exp.get('status', 'skipped'),
                        'improvement': 'N/A',
                        'summary': exp.get('reason', 'No code generation attempted'),
                        'deployment_status': 'not_deployed',
                        'github_committed': False,
                        'github_commit_hash': None
                    }
                    break
        
        # Write to DynamoDB experiments table
        experiments_table = dynamodb.Table('ai-video-codec-experiments')
        item_data = {
            'experiment_id': results['experiment_id'],
            'timestamp': int(time.time()),  # Unix timestamp as number
            'timestamp_iso': results['timestamp'],  # ISO format for readability
            'experiments': json.dumps(results['experiments']),
            'status': 'completed',
            's3_key': key
        }
        
        # Add code evolution info if available
        if code_evolution_info:
            item_data['code_evolution'] = json.dumps(code_evolution_info)
        
        experiments_table.put_item(Item=item_data)
        logger.info(f"Experiment logged to DynamoDB: {results['experiment_id']}")
        
        # Write individual metrics to DynamoDB metrics table
        metrics_table = dynamodb.Table('ai-video-codec-metrics')
        for exp in results['experiments']:
            if exp.get('status') == 'completed':
                metrics_table.put_item(
                    Item={
                        'metric_id': f"{results['experiment_id']}_{exp['experiment_type']}",
                        'experiment_id': results['experiment_id'],
                        'timestamp': int(time.time()),  # Unix timestamp as number
                        'timestamp_iso': exp['timestamp'],  # ISO format for readability
                        'experiment_type': exp['experiment_type'],
                        'metrics': json.dumps(exp.get('real_metrics', exp.get('neural_networks', {})))
                    }
                )
        logger.info(f"Metrics logged to DynamoDB")
        
        return True
    except Exception as e:
        logger.error(f"Failed to upload real results: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_llm_pre_analysis(past_experiments):
    """Run LLM analysis BEFORE starting experiment."""
    logger.info("🤖 Running LLM pre-experiment analysis...")
    
    try:
        from agents.llm_experiment_planner import LLMExperimentPlanner
        
        planner = LLMExperimentPlanner()
        
        # Get hypothesis and plan for next experiment
        if past_experiments:
            # Convert past experiments to format expected by LLM
            formatted_experiments = []
            for exp in past_experiments:
                exp_copy = exp.copy()
                # Ensure experiments field is a string
                if not isinstance(exp_copy.get('experiments'), str):
                    exp_copy['experiments'] = json.dumps(exp_copy.get('experiments', []))
                formatted_experiments.append(exp_copy)
            
            # Get LLM analysis of past experiments
            analysis = planner.get_llm_analysis(formatted_experiments)
            
            if analysis:
                logger.info(f"💡 LLM Hypothesis: {analysis.get('hypothesis', 'N/A')[:100]}...")
                logger.info(f"🎯 Expected improvement: {analysis.get('expected_bitrate_mbps', 'N/A')} Mbps")
                
                # NEW: Generate compression code based on analysis
                logger.info("🔧 Generating new compression algorithm...")
                generated_code = planner.generate_compression_code(analysis)
                if generated_code:
                    analysis['generated_code'] = generated_code
                    logger.info(f"✅ Generated {len(generated_code['code'])} chars of compression code")
                else:
                    logger.warning("Code generation failed - will use default algorithms")
                
                return analysis
            else:
                logger.warning("No LLM analysis available")
                return None
        else:
            logger.info("No past experiments to analyze (baseline run)")
            return None
            
    except Exception as e:
        logger.error(f"LLM pre-analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_llm_post_analysis(experiment_result, past_experiments):
    """Run LLM analysis AFTER experiment completes."""
    logger.info("🤖 Running LLM post-experiment analysis...")
    
    try:
        from agents.llm_experiment_planner import LLMExperimentPlanner
        
        planner = LLMExperimentPlanner()
        
        # Prepare experiment data for analysis
        # Ensure experiments field is a string
        exp_data = experiment_result.copy()
        if not isinstance(exp_data.get('experiments'), str):
            exp_data['experiments'] = json.dumps(exp_data.get('experiments', []))
        
        # Format past experiments similarly
        formatted_past = []
        for exp in past_experiments:
            exp_copy = exp.copy()
            if not isinstance(exp_copy.get('experiments'), str):
                exp_copy['experiments'] = json.dumps(exp_copy.get('experiments', []))
            formatted_past.append(exp_copy)
        
        # Analyze the just-completed experiment with all past ones
        all_experiments = [exp_data] + formatted_past
        analysis = planner.get_llm_analysis(all_experiments)
        
        if analysis:
            logger.info(f"📊 LLM Analysis complete")
            logger.info(f"   Root cause: {analysis.get('root_cause', 'N/A')[:100]}...")
            logger.info(f"   Next hypothesis: {analysis.get('hypothesis', 'N/A')[:100]}...")
            
            # Store reasoning in DynamoDB using planner's method
            planner.log_reasoning(analysis, exp_data.get('experiment_id', 'unknown'))
            logger.info(f"   ✅ Reasoning stored in DynamoDB")
            
            return analysis
        else:
            logger.warning("No reasoning generated")
            return None
            
    except Exception as e:
        logger.error(f"LLM post-analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run real AI codec experiments."""
    logger.info("🎬 REAL AI Video Codec Experiment Starting...")
    
    all_results = {
        'experiment_id': f"real_exp_{int(time.time())}",
        'timestamp': datetime.utcnow().isoformat(),
        'experiments': []
    }
    
    try:
        # Fetch past experiments for LLM context
        logger.info("📚 Fetching past experiments for LLM context...")
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        experiments_table = dynamodb.Table('ai-video-codec-experiments')
        past_response = experiments_table.scan(Limit=20)
        past_experiments = past_response.get('Items', [])
        past_experiments.sort(key=lambda x: x.get('timestamp', 0))
        logger.info(f"Found {len(past_experiments)} past experiments")
        
        # LLM PRE-ANALYSIS: Generate hypothesis before starting
        logger.info("=" * 50)
        logger.info("LLM PRE-EXPERIMENT ANALYSIS")
        logger.info("=" * 50)
        pre_analysis = run_llm_pre_analysis(past_experiments)
        
        if pre_analysis:
            all_results['pre_analysis'] = pre_analysis
        
        logger.info("=" * 50)
        # OPTION C: Run ONLY LLM-GENERATED CODE experiment
        # Compare against: (1) HEVC baseline (2) Previous LLM iteration
        logger.info("=" * 50)
        logger.info("LLM AUTONOMOUS CODE EVOLUTION")
        logger.info("=" * 50)
        llm_code_results = run_llm_generated_experiment(llm_config=pre_analysis)
        all_results['experiments'].append(llm_code_results)
        
        # Add baseline comparison metadata
        # The LLM can use neural networks (torch) in its generated code
        # The adaptive_codec_agent will compare performance against previous iterations
        logger.info("📊 Baseline: HEVC 10 Mbps (1080p@30fps)")
        logger.info("📈 Previous best: " + 
                   (f"{llm_code_results.get('evolution', {}).get('metrics', {}).get('bitrate_mbps', 'N/A')} Mbps"
                    if llm_code_results.get('evolution', {}).get('adopted') 
                    else "No previous version"))
        
        # Upload results FIRST
        logger.info("=" * 50)
        logger.info("UPLOADING RESULTS")
        logger.info("=" * 50)
        upload_success = upload_real_results(all_results)
        
        if not upload_success:
            logger.error("❌ Failed to upload results")
            return 1
        
        # LLM POST-ANALYSIS: Analyze what happened
        logger.info("=" * 50)
        logger.info("LLM POST-EXPERIMENT ANALYSIS")
        logger.info("=" * 50)
        post_analysis = run_llm_post_analysis(all_results, past_experiments)
        
        if post_analysis:
            logger.info("✅ Post-experiment analysis complete and stored")
        else:
            logger.warning("⚠️  Post-experiment analysis skipped")
        
        logger.info("🎉 REAL AI Codec Experiment completed successfully!")
        logger.info(f"Results: {json.dumps(all_results, indent=2)}")
        return 0
            
    except Exception as e:
        logger.error(f"❌ Real experiment failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
