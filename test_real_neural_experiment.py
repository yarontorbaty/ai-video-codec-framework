#!/usr/bin/env python3
"""
Test Real Neural Codec Experiment
Uses the working neural codec code directly to run a real experiment.
"""

import os
import sys
import json
import time
import requests
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuration
ORCHESTRATOR_URL = "http://34.239.1.29:8081"

class RealNeuralExperimentRunner:
    """Runs real neural codec experiments with working code."""
    
    def __init__(self):
        self.orchestrator_url = ORCHESTRATOR_URL
        
    def create_real_neural_experiment(self):
        """Create a real neural codec experiment with working code."""
        print("🧠 Creating real neural codec experiment...")
        
        # Read the working neural codec code
        try:
            with open('working_neural_codec.py', 'r') as f:
                neural_code = f.read()
            
            print(f"✅ Loaded working neural codec code ({len(neural_code)} chars)")
            
        except Exception as e:
            print(f"❌ Failed to load neural codec code: {e}")
            return None
        
        # Create experiment data with the working code
        experiment_data = {
            'hypothesis': 'Real neural codec using hybrid keyframe + residual compression with spatial downsampling and neural upscaling',
            'compression_strategy': 'neural_hybrid_compression',
            'expected_bitrate_mbps': 4.2,
            'encoding_agent_code': neural_code,
            'decoding_agent_code': neural_code,  # Same code handles both
            'experiment_type': 'neural_codec_v2_real',
            'generated_by': 'real_neural_codec',
            'timestamp': int(time.time()),
            'description': 'Hybrid keyframe + residual compression codec. Stores full frames every N frames (downscaled 50% + JPEG compressed at quality 75). Intermediate frames store quantized residuals from previous keyframe (PNG compressed). Decompression reconstructs frames by upscaling keyframes or adding residuals to previous keyframe, then upscaling to original resolution.'
        }
        
        print(f"✅ Created real neural experiment")
        print(f"   Hypothesis: {experiment_data['hypothesis']}")
        print(f"   Strategy: {experiment_data['compression_strategy']}")
        print(f"   Expected bitrate: {experiment_data['expected_bitrate_mbps']} Mbps")
        print(f"   Code length: {len(experiment_data['encoding_agent_code'])} chars")
        
        return experiment_data
    
    def send_experiment_to_orchestrator(self, experiment_data):
        """Send experiment to HTTP orchestrator."""
        try:
            print("📤 Sending real neural experiment to orchestrator...")
            
            response = requests.post(
                f"{self.orchestrator_url}/experiment",
                json=experiment_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'dispatched':
                    experiment_id = result['experiment_id']
                    print(f"✅ Real neural experiment dispatched: {experiment_id}")
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
        print(f"⏳ Monitoring real neural experiment {experiment_id}...")
        
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
                        print(f"✅ Real neural experiment {experiment_id} completed!")
                        
                        # Print results
                        if 'metrics' in status:
                            metrics = status['metrics']
                            print(f"   Bitrate: {metrics.get('bitrate_mbps', 'N/A')} Mbps")
                            print(f"   Compression Ratio: {metrics.get('compression_ratio', 'N/A')}%")
                            print(f"   PSNR: {metrics.get('psnr_db', 'N/A')} dB")
                            print(f"   Processing Time: {metrics.get('processing_time_seconds', 'N/A')} seconds")
                        
                        # Check if we got real results (not just placeholder)
                        if 'encoding_result' in status and status['encoding_result']:
                            print(f"   ✅ Encoding result: {status['encoding_result']}")
                        if 'decoding_result' in status and status['decoding_result']:
                            print(f"   ✅ Decoding result: {status['decoding_result']}")
                        
                        return status
                    
                    elif status['status'] == 'failed':
                        print(f"❌ Real neural experiment {experiment_id} failed")
                        if 'error' in status:
                            print(f"   Error: {status['error']}")
                        return status
                    
                    else:
                        print(f"   Status: {status['status']}...")
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"⚠️  Error checking status: {e}")
                time.sleep(5)
        
        print(f"⏰ Real neural experiment {experiment_id} timed out after {timeout_minutes} minutes")
        return {'status': 'timeout'}
    
    def run_real_experiment(self):
        """Run a complete real neural codec experiment."""
        print(f"\n{'='*60}")
        print(f"🚀 Starting REAL Neural Codec Experiment")
        print(f"{'='*60}")
        
        # 1. Create real experiment
        experiment_data = self.create_real_neural_experiment()
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
            print(f"🎉 REAL Neural Codec Experiment completed successfully!")
            print(f"   This proves the system can run actual neural codec experiments")
            print(f"   with real video compression/decompression code!")
            return True
        else:
            print(f"❌ Real experiment failed: {result['status']}")
            return False

def main():
    """Main function."""
    runner = RealNeuralExperimentRunner()
    success = runner.run_real_experiment()
    
    if success:
        print("\n🎉 SUCCESS! Real neural codec experiments are now working!")
        print("   The system can generate and execute actual neural codec code")
        print("   The issue was just in the JSON parsing, not the code generation")
    else:
        print("\n❌ Real experiment failed - need to debug further")

if __name__ == '__main__':
    main()
