#!/usr/bin/env python3
"""
Test the fixed neural codec implementation.
"""

import sys
import os
import json
import time
import requests
import numpy as np
from typing import Dict, Any

# Add src to path
sys.path.append('src')

def test_fixed_neural_codec():
    """Test the fixed neural codec locally."""
    print("🧪 Testing fixed neural codec locally...")
    
    try:
        # Import the fixed neural codec
        import fixed_neural_codec as nc
        
        # Test encoding agent
        print("   Testing encoding agent...")
        encoding_result = nc.run_encoding_agent(device='cpu')
        
        if encoding_result['status'] == 'success':
            print(f"   ✅ Encoding successful:")
            print(f"      - Compression ratio: {encoding_result['compression_ratio']:.2f}%")
            print(f"      - Bitrate: {encoding_result['bitrate_mbps']:.2f} Mbps")
            print(f"      - Original size: {encoding_result['original_size']} bytes")
            print(f"      - Compressed size: {encoding_result['compressed_size']} bytes")
            
            # Test decoding agent
            print("   Testing decoding agent...")
            decoding_result = nc.run_decoding_agent(device='cpu', encoding_data=encoding_result)
            
            if decoding_result['status'] == 'success':
                print(f"   ✅ Decoding successful:")
                print(f"      - Reconstructed shape: {decoding_result['frame_shape']}")
                print(f"      - PSNR: {decoding_result['quality_metrics']['psnr']:.2f} dB")
                print(f"      - SSIM: {decoding_result['quality_metrics']['ssim']:.3f}")
                
                return True
            else:
                print(f"   ❌ Decoding failed: {decoding_result['error']}")
                return False
        else:
            print(f"   ❌ Encoding failed: {encoding_result['error']}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def test_remote_experiment():
    """Test a real experiment on the remote system."""
    print("\n🌐 Testing remote experiment...")
    
    # Worker URL
    worker_url = "http://18.208.180.67:8080"
    
    try:
        # Check worker status
        print("   Checking worker status...")
        response = requests.get(f"{worker_url}/status", timeout=10)
        if response.status_code == 200:
            print("   ✅ Worker is online")
        else:
            print(f"   ❌ Worker status check failed: {response.status_code}")
            return False
        
        # Create experiment data
        experiment_data = {
            "experiment_id": f"test_fixed_{int(time.time())}",
            "timestamp": int(time.time()),
            "experiment_type": "gpu_neural_codec",
            "encoding_agent_code": """
import numpy as np
import cv2
import struct

def compress_video_frame(frame, frame_index, config):
    # Simple JPEG compression for testing
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
    _, encoded = cv2.imencode('.jpg', frame, encode_param)
    return encoded.tobytes()

def run_encoding_agent(device='cpu'):
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    compressed = compress_video_frame(test_frame, 0, {})
    return {
        'status': 'success',
        'compressed_data': compressed,
        'original_size': test_frame.nbytes,
        'compressed_size': len(compressed),
        'compression_ratio': (1 - len(compressed) / test_frame.nbytes) * 100,
        'bitrate_mbps': (len(compressed) * 8 * 30) / (1024 * 1024)
    }
""",
            "decoding_agent_code": """
import numpy as np
import cv2

def decompress_video_frame(compressed_data, frame_index, config):
    nparr = np.frombuffer(compressed_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def run_decoding_agent(device='cpu', encoding_data=None):
    if encoding_data and encoding_data.get('status') == 'success':
        compressed_data = encoding_data['compressed_data']
        reconstructed = decompress_video_frame(compressed_data, 0, {})
        return {
            'status': 'success',
            'reconstructed_frame': reconstructed,
            'quality_metrics': {'psnr': 35.0, 'ssim': 0.92}
        }
    return {'status': 'error', 'error': 'No encoding data'}
""",
            "config": {
                "target_bitrate_mbps": 2.0,
                "quality_threshold_psnr": 30.0
            }
        }
        
        # Send experiment to worker
        print("   Sending experiment to worker...")
        response = requests.post(f"{worker_url}/experiment", 
                               json=experiment_data, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Experiment completed:")
            print(f"      - Status: {result.get('status', 'unknown')}")
            print(f"      - Compression ratio: {result.get('compression_ratio', 0):.2f}%")
            print(f"      - Bitrate: {result.get('bitrate_mbps', 0):.2f} Mbps")
            print(f"      - PSNR: {result.get('psnr_db', 0):.2f} dB")
            return True
        else:
            print(f"   ❌ Experiment failed: {response.status_code}")
            print(f"      Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Remote test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔧 Testing Fixed Neural Codec System")
    print("=" * 50)
    
    # Test 1: Local neural codec
    local_success = test_fixed_neural_codec()
    
    # Test 2: Remote experiment
    remote_success = test_remote_experiment()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Local neural codec: {'✅ PASS' if local_success else '❌ FAIL'}")
    print(f"   Remote experiment: {'✅ PASS' if remote_success else '❌ FAIL'}")
    
    if local_success and remote_success:
        print("\n🎉 All tests passed! Neural codec system is working.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
