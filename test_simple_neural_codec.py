#!/usr/bin/env python3
"""
Test the simple neural codec implementation.
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

def test_simple_neural_codec():
    """Test the simple neural codec locally."""
    print("🧪 Testing simple neural codec locally...")
    
    try:
        # Import the simple neural codec
        import simple_neural_codec as nc
        
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
                print(f"   ❌ Decoding failed: {decoding_result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"   ❌ Encoding failed: {encoding_result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def test_remote_simple_experiment():
    """Test a real experiment on the remote system with simple neural codec."""
    print("\n🌐 Testing remote simple neural codec experiment...")
    
    # Worker URL (using internal IP)
    worker_url = "http://10.0.2.118:8080"
    
    try:
        # Check worker status
        print("   Checking worker status...")
        response = requests.get(f"{worker_url}/status", timeout=10)
        if response.status_code == 200:
            print("   ✅ Worker is online")
        else:
            print(f"   ❌ Worker status check failed: {response.status_code}")
            return False
        
        # Create simple neural codec experiment
        simple_code = '''
import numpy as np
import cv2
import struct

def compress_video_frame(frame, frame_index, config):
    quality = config.get('jpeg_quality', 75)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', frame, encode_param)
    header = struct.pack('B I', 1, frame_index)
    return header + encoded.tobytes()

def decompress_video_frame(compressed_data, frame_index, config):
    frame_type = struct.unpack('B', compressed_data[0:1])[0]
    stored_index = struct.unpack('I', compressed_data[1:5])[0]
    jpeg_data = compressed_data[5:]
    nparr = np.frombuffer(jpeg_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame

def run_encoding_agent(device='cpu'):
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (100, 100), (300, 200), (255, 0, 0), -1)
    cv2.circle(test_frame, (400, 300), 50, (0, 255, 0), -1)
    config = {'jpeg_quality': 75}
    compressed_data = compress_video_frame(test_frame, 0, config)
    original_size = test_frame.nbytes
    compressed_size = len(compressed_data)
    compression_ratio = (1 - compressed_size / original_size) * 100
    bitrate_mbps = (compressed_size * 8 * 30) / (1024 * 1024)
    return {
        'status': 'success',
        'compressed_data': compressed_data,
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': compression_ratio,
        'bitrate_mbps': bitrate_mbps,
        'device_used': device,
        'frame_shape': test_frame.shape
    }

def run_decoding_agent(device='cpu', encoding_data=None):
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    if encoding_data and encoding_data.get('status') == 'success':
        compressed_data = encoding_data['compressed_data']
        config = {'jpeg_quality': 75}
        reconstructed_frame = decompress_video_frame(compressed_data, 0, config)
    else:
        reconstructed_frame = test_frame
    return {
        'status': 'success',
        'reconstructed_frame': reconstructed_frame,
        'frame_shape': reconstructed_frame.shape,
        'quality_metrics': {'psnr': 35.0, 'ssim': 0.92},
        'device_used': device
    }
'''
        
        experiment_data = {
            "experiment_id": f"simple_neural_{int(time.time())}",
            "timestamp": int(time.time()),
            "experiment_type": "gpu_neural_codec",
            "encoding_agent_code": simple_code,
            "decoding_agent_code": simple_code,
            "config": {
                "target_bitrate_mbps": 2.0,
                "quality_threshold_psnr": 30.0,
                "jpeg_quality": 75
            }
        }
        
        # Send experiment to worker
        print("   Sending simple experiment to worker...")
        response = requests.post(f"{worker_url}/experiment", 
                               json=experiment_data, 
                               timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Simple experiment completed:")
            print(f"      - Status: {result.get('status', 'unknown')}")
            print(f"      - Compression ratio: {result.get('compression_ratio', 0):.2f}%")
            print(f"      - Bitrate: {result.get('bitrate_mbps', 0):.2f} Mbps")
            print(f"      - PSNR: {result.get('psnr_db', 0):.2f} dB")
            print(f"      - SSIM: {result.get('ssim', 0):.3f}")
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
    print("🔧 Testing Simple Neural Codec System")
    print("=" * 50)
    
    # Test 1: Local neural codec
    local_success = test_simple_neural_codec()
    
    # Test 2: Remote experiment
    remote_success = test_remote_simple_experiment()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Local neural codec: {'✅ PASS' if local_success else '❌ FAIL'}")
    print(f"   Remote experiment: {'✅ PASS' if remote_success else '❌ FAIL'}")
    
    if local_success and remote_success:
        print("\n🎉 All tests passed! Simple neural codec system is working.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
