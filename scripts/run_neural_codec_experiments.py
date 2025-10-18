#!/usr/bin/env python3
"""
Run Real Neural Codec Experiments
Uses LLM to generate actual neural codec experiments and sends them to HTTP system.
"""

import os
import sys
import json
import time
import requests
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.llm_experiment_planner import LLMExperimentPlanner

# Configuration
ORCHESTRATOR_URL = "http://34.239.1.29:8081"
EXPERIMENT_COUNT = 3  # Number of experiments to run

class NeuralCodecExperimentRunner:
    """Runs real neural codec experiments using LLM planning."""
    
    def __init__(self):
        self.orchestrator_url = ORCHESTRATOR_URL
        self.llm_planner = LLMExperimentPlanner()
        
    def check_orchestrator_health(self):
        """Check if orchestrator is healthy."""
        try:
            response = requests.get(f"{self.orchestrator_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Orchestrator healthy: {data.get('available_workers', 0)} workers available")
                return True
            else:
                print(f"❌ Orchestrator unhealthy: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Orchestrator unreachable: {e}")
            return False
    
    def generate_neural_experiment(self, experiment_number):
        """Generate a real neural codec experiment using LLM."""
        print(f"\n🧠 Generating neural codec experiment #{experiment_number}...")
        
        try:
            # Use LLM to generate experiment
            analysis = self.llm_planner.plan_next_experiment()
            
            if not analysis:
                print("❌ LLM failed to generate experiment")
                return None
            
            # Extract experiment data
            experiment_data = {
                'hypothesis': analysis.get('hypothesis', 'Neural codec compression experiment'),
                'compression_strategy': analysis.get('compression_strategy', 'neural_compression'),
                'expected_bitrate_mbps': analysis.get('expected_bitrate_mbps', 1.0),
                'encoding_agent_code': analysis.get('encoding_agent_code', ''),
                'decoding_agent_code': analysis.get('decoding_agent_code', ''),
                'experiment_type': 'neural_codec_v2',
                'generated_by': 'llm_planner',
                'timestamp': int(time.time())
            }
            
            print(f"✅ Generated experiment: {experiment_data['hypothesis']}")
            print(f"   Strategy: {experiment_data['compression_strategy']}")
            print(f"   Expected bitrate: {experiment_data['expected_bitrate_mbps']} Mbps")
            
            return experiment_data
            
        except Exception as e:
            print(f"❌ Failed to generate experiment: {e}")
            return None
    
    def send_experiment_to_orchestrator(self, experiment_data):
        """Send experiment to HTTP orchestrator."""
        try:
            print("📤 Sending experiment to orchestrator...")
            
            response = requests.post(
                f"{self.orchestrator_url}/experiment",
                json=experiment_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'dispatched':
                    experiment_id = result['experiment_id']
                    print(f"✅ Experiment dispatched: {experiment_id}")
                    return experiment_id
                else:
                    print(f"❌ Experiment not dispatched: {result}")
                    return None
            else:
                print(f"❌ Failed to send experiment: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error sending experiment: {e}")
            return None
    
    def monitor_experiment(self, experiment_id, timeout_minutes=10):
        """Monitor experiment until completion or timeout."""
        print(f"⏳ Monitoring experiment {experiment_id}...")
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            try:
                response = requests.get(
                    f"{self.orchestrator_url}/experiment/{experiment_id}/status",
                    timeout=10
                )
                
                if response.status_code == 200:
                    status = response.json()
                    
                    if status['status'] == 'completed':
                        print(f"✅ Experiment {experiment_id} completed!")
                        
                        # Print results
                        if 'metrics' in status:
                            metrics = status['metrics']
                            print(f"   Bitrate: {metrics.get('bitrate_mbps', 'N/A')} Mbps")
                            print(f"   Compression Ratio: {metrics.get('compression_ratio', 'N/A')}%")
                            print(f"   PSNR: {metrics.get('psnr_db', 'N/A')} dB")
                        
                        return status
                    
                    elif status['status'] == 'failed':
                        print(f"❌ Experiment {experiment_id} failed")
                        if 'error' in status:
                            print(f"   Error: {status['error']}")
                        return status
                    
                    else:
                        print(f"   Status: {status['status']}...")
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"⚠️  Error checking status: {e}")
                time.sleep(5)
        
        print(f"⏰ Experiment {experiment_id} timed out after {timeout_minutes} minutes")
        return {'status': 'timeout'}
    
    def run_experiment_cycle(self, experiment_number):
        """Run a complete experiment cycle."""
        print(f"\n{'='*60}")
        print(f"🚀 Starting Neural Codec Experiment #{experiment_number}")
        print(f"{'='*60}")
        
        # 1. Generate experiment
        experiment_data = self.generate_neural_experiment(experiment_number)
        if not experiment_data:
            return False
        
        # 2. Send to orchestrator
        experiment_id = self.send_experiment_to_orchestrator(experiment_data)
        if not experiment_id:
            return False
        
        # 3. Monitor until completion
        result = self.monitor_experiment(experiment_id)
        
        # 4. Report results
        if result['status'] == 'completed':
            print(f"🎉 Experiment #{experiment_number} completed successfully!")
            return True
        else:
            print(f"❌ Experiment #{experiment_number} failed: {result['status']}")
            return False
    
    def run_multiple_experiments(self, count=EXPERIMENT_COUNT):
        """Run multiple neural codec experiments."""
        print(f"🧪 Starting Neural Codec Experiment Series")
        print(f"   Orchestrator: {self.orchestrator_url}")
        print(f"   Experiments: {count}")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check orchestrator health
        if not self.check_orchestrator_health():
            print("❌ Cannot proceed - orchestrator not healthy")
            return
        
        success_count = 0
        
        for i in range(1, count + 1):
            try:
                success = self.run_experiment_cycle(i)
                if success:
                    success_count += 1
                
                # Wait between experiments
                if i < count:
                    print(f"\n⏳ Waiting 30 seconds before next experiment...")
                    time.sleep(30)
                    
            except KeyboardInterrupt:
                print(f"\n⏹️  Stopped by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error in experiment #{i}: {e}")
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"📊 Experiment Series Complete")
        print(f"{'='*60}")
        print(f"   Total experiments: {count}")
        print(f"   Successful: {success_count}")
        print(f"   Success rate: {(success_count/count)*100:.1f}%")
        
        if success_count > 0:
            print(f"🎉 Neural codec experiments are working!")
        else:
            print(f"❌ All experiments failed - check system health")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Neural Codec Experiments')
    parser.add_argument('--count', type=int, default=EXPERIMENT_COUNT, 
                       help=f'Number of experiments to run (default: {EXPERIMENT_COUNT})')
    parser.add_argument('--once', action='store_true', 
                       help='Run single experiment and exit')
    
    args = parser.parse_args()
    
    runner = NeuralCodecExperimentRunner()
    
    if args.once:
        runner.run_experiment_cycle(1)
    else:
        runner.run_multiple_experiments(args.count)

if __name__ == '__main__':
    main()
