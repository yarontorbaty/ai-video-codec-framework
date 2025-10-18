#!/usr/bin/env python3
"""
Test HTTP-based Neural Codec Pipeline
Tests the new HTTP communication between orchestrator and worker.
"""

import requests
import json
import time
import sys

def test_worker_health(worker_url):
    """Test worker health endpoint."""
    try:
        response = requests.get(f"{worker_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Worker health: {data['status']}")
            print(f"   Device: {data.get('device', 'unknown')}")
            print(f"   Jobs processed: {data.get('jobs_processed', 0)}")
            return True
        else:
            print(f"❌ Worker health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Worker not reachable: {e}")
        return False

def test_orchestrator_health(orchestrator_url):
    """Test orchestrator health endpoint."""
    try:
        response = requests.get(f"{orchestrator_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Orchestrator health: {data['status']}")
            print(f"   Available workers: {len(data.get('available_workers', []))}")
            return True
        else:
            print(f"❌ Orchestrator health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Orchestrator not reachable: {e}")
        return False

def test_experiment_flow(orchestrator_url):
    """Test complete experiment flow."""
    print("\n🎯 Testing experiment flow...")
    
    # Create test experiment
    experiment_data = {
        'hypothesis': 'Test HTTP-based neural codec',
        'compression_strategy': 'semantic_compression',
        'expected_bitrate_mbps': 1.0,
        'encoding_agent_code': '''
def run_encoding_agent(device='cpu'):
    """Test encoding agent."""
    import torch
    return {
        'compressed_data': 'test_encoding_result',
        'semantic_description': 'Test video content',
        'device_used': device
    }
''',
        'decoding_agent_code': '''
def run_decoding_agent(device='cpu', encoding_data=None):
    """Test decoding agent."""
    import torch
    return {
        'reconstructed_video': 'test_decoding_result',
        'quality_metrics': {'psnr': 45.0, 'ssim': 0.98},
        'device_used': device
    }
'''
    }
    
    try:
        # Send experiment to orchestrator
        print("📤 Sending experiment to orchestrator...")
        response = requests.post(
            f"{orchestrator_url}/experiment",
            json=experiment_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Experiment accepted: {result['status']}")
            print(f"   Experiment ID: {result['experiment_id']}")
            
            if result['status'] == 'dispatched':
                experiment_id = result['experiment_id']
                print(f"   Worker URL: {result['worker_url']}")
                
                # Wait for completion
                print("⏳ Waiting for completion...")
                for i in range(30):  # Wait up to 30 seconds
                    time.sleep(1)
                    
                    status_response = requests.get(
                        f"{orchestrator_url}/experiment/{experiment_id}/status",
                        timeout=5
                    )
                    
                    if status_response.status_code == 200:
                        status = status_response.json()
                        if status['status'] == 'completed':
                            print("✅ Experiment completed!")
                            print(f"   Metrics: {status.get('metrics', {})}")
                            return True
                        elif status['status'] == 'failed':
                            print(f"❌ Experiment failed: {status.get('error', 'Unknown error')}")
                            return False
                        else:
                            print(f"   Status: {status['status']}...")
                    else:
                        print(f"⚠️  Status check failed: {status_response.status_code}")
                
                print("⏰ Timeout waiting for completion")
                return False
            else:
                print(f"❌ Experiment not dispatched: {result}")
                return False
        else:
            print(f"❌ Failed to send experiment: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Experiment test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 HTTP Neural Codec Pipeline Test")
    print("=" * 50)
    
    # Configuration
    worker_url = "http://localhost:8080"
    orchestrator_url = "http://localhost:8081"
    
    # Test 1: Worker health
    print("\n1️⃣ Testing worker health...")
    worker_healthy = test_worker_health(worker_url)
    
    # Test 2: Orchestrator health
    print("\n2️⃣ Testing orchestrator health...")
    orchestrator_healthy = test_orchestrator_health(orchestrator_url)
    
    if not worker_healthy or not orchestrator_healthy:
        print("\n❌ Health checks failed - cannot proceed with experiment test")
        print("Make sure both worker and orchestrator are running:")
        print(f"  Worker: {worker_url}")
        print(f"  Orchestrator: {orchestrator_url}")
        sys.exit(1)
    
    # Test 3: Complete experiment flow
    print("\n3️⃣ Testing complete experiment flow...")
    experiment_success = test_experiment_flow(orchestrator_url)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Worker Health: {'✅' if worker_healthy else '❌'}")
    print(f"   Orchestrator Health: {'✅' if orchestrator_healthy else '❌'}")
    print(f"   Experiment Flow: {'✅' if experiment_success else '❌'}")
    
    if worker_healthy and orchestrator_healthy and experiment_success:
        print("\n🎉 All tests passed! HTTP pipeline is working.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)

if __name__ == '__main__':
    main()
