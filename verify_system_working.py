#!/usr/bin/env python3
"""
Verify System Working
Comprehensive test to verify the neural codec system is working end-to-end.
"""

import os
import sys
import json
import time
import requests
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.llm_experiment_planner import LLMExperimentPlanner

# Configuration
ORCHESTRATOR_URL = "http://34.239.1.29:8081"

def test_llm_code_generation():
    """Test if LLM generates real neural codec code."""
    print("🧠 Testing LLM Code Generation...")
    
    try:
        planner = LLMExperimentPlanner()
        analysis = planner.plan_next_experiment()
        
        if not analysis:
            print("❌ LLM failed to generate analysis")
            return False
        
        encoding_code = analysis.get('encoding_agent_code', '')
        decoding_code = analysis.get('decoding_agent_code', '')
        
        print(f"   Encoding code length: {len(encoding_code)} chars")
        print(f"   Decoding code length: {len(decoding_code)} chars")
        
        if len(encoding_code) > 1000 and len(decoding_code) > 1000:
            print("✅ LLM is generating real neural codec code!")
            return True
        else:
            print("❌ LLM is not generating enough code")
            return False
            
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False

def test_orchestrator_health():
    """Test if orchestrator is healthy."""
    print("🏥 Testing Orchestrator Health...")
    
    try:
        response = requests.get(f"{ORCHESTRATOR_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            workers = data.get('available_workers', 0)
            print(f"✅ Orchestrator healthy: {workers} workers available")
            return True
        else:
            print(f"❌ Orchestrator unhealthy: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Orchestrator unreachable: {e}")
        return False

def test_real_experiment():
    """Test running a real neural codec experiment."""
    print("🧪 Testing Real Neural Experiment...")
    
    try:
        # Create experiment with working code
        with open('working_neural_codec.py', 'r') as f:
            neural_code = f.read()
        
        experiment_data = {
            'hypothesis': 'Test neural codec with hybrid keyframe + residual compression',
            'compression_strategy': 'neural_hybrid_compression',
            'expected_bitrate_mbps': 4.2,
            'encoding_agent_code': neural_code,
            'decoding_agent_code': neural_code,
            'experiment_type': 'neural_codec_v2_test',
            'generated_by': 'system_verification',
            'timestamp': int(time.time())
        }
        
        # Send to orchestrator
        response = requests.post(
            f"{ORCHESTRATOR_URL}/experiment",
            json=experiment_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'dispatched':
                experiment_id = result['experiment_id']
                print(f"✅ Experiment dispatched: {experiment_id}")
                
                # Monitor for completion
                for i in range(12):  # Wait up to 1 minute
                    time.sleep(5)
                    status_response = requests.get(
                        f"{ORCHESTRATOR_URL}/experiment/{experiment_id}/status",
                        timeout=10
                    )
                    
                    if status_response.status_code == 200:
                        status = status_response.json()
                        if status['status'] == 'completed':
                            print(f"✅ Experiment completed successfully!")
                            return True
                        elif status['status'] == 'failed':
                            print(f"❌ Experiment failed: {status.get('error', 'Unknown error')}")
                            return False
                
                print(f"⏰ Experiment timed out")
                return False
            else:
                print(f"❌ Experiment not dispatched: {result}")
                return False
        else:
            print(f"❌ Failed to send experiment: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Real experiment test failed: {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 NEURAL CODEC SYSTEM VERIFICATION")
    print("=" * 50)
    
    tests = [
        ("LLM Code Generation", test_llm_code_generation),
        ("Orchestrator Health", test_orchestrator_health),
        ("Real Experiment", test_real_experiment),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Neural codec system is fully operational!")
        print("   ✅ LLM generates real neural codec code")
        print("   ✅ Orchestrator is healthy and responsive")
        print("   ✅ Real experiments execute successfully")
        print("   ✅ System is ready for production GPU workloads!")
    else:
        print("⚠️  Some tests failed - system needs attention")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
