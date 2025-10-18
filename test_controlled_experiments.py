#!/usr/bin/env python3
"""
Send controlled test experiments to GPU worker for validation.
"""
import requests
import json
import time

WORKER_URL = "http://10.0.2.118:8080"

# Test 1: Simple JPEG compression
test1_encoding = '''
import torch
import torch.nn as nn
import cv2
import numpy as np

def run_encoding_agent(frame, device='cpu'):
    """Simple JPEG-like compression."""
    # Compress using JPEG quality 50
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    result, encoded = cv2.imencode('.jpg', frame, encode_param)
    
    return {
        'status': 'success',
        'compressed_data': encoded.tobytes(),
        'compression_method': 'JPEG-50'
    }
'''

test1_decoding = '''
import cv2
import numpy as np

def run_decoding_agent(device, encoding_data, compressed_path, output_path):
    """Decode JPEG compressed data."""
    # Decode from bytes
    compressed_data = encoding_data.get('compressed_data')
    nparr = np.frombuffer(compressed_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Save as image
    cv2.imwrite(output_path, frame)
    
    return {
        'status': 'success',
        'reconstructed_frame': frame
    }
'''

# Test 2: Higher quality JPEG
test2_encoding = '''
import torch
import torch.nn as nn
import cv2
import numpy as np

def run_encoding_agent(frame, device='cpu'):
    """Higher quality JPEG compression."""
    # Compress using JPEG quality 80
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    result, encoded = cv2.imencode('.jpg', frame, encode_param)
    
    return {
        'status': 'success',
        'compressed_data': encoded.tobytes(),
        'compression_method': 'JPEG-80'
    }
'''

test2_decoding = '''
import cv2
import numpy as np

def run_decoding_agent(device, encoding_data, compressed_path, output_path):
    """Decode JPEG compressed data."""
    # Decode from bytes
    compressed_data = encoding_data.get('compressed_data')
    nparr = np.frombuffer(compressed_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Save as image
    cv2.imwrite(output_path, frame)
    
    return {
        'status': 'success',
        'reconstructed_frame': frame
    }
'''

def send_experiment(experiment_id, encoding_code, decoding_code):
    """Send experiment to worker."""
    print(f"\n📤 Sending experiment: {experiment_id}")
    
    payload = {
        'experiment_id': experiment_id,
        'encoding_code': encoding_code,
        'decoding_code': decoding_code,
        'timestamp': int(time.time())
    }
    
    try:
        response = requests.post(f"{WORKER_URL}/experiment", json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        print(f"✅ Experiment completed: {result.get('status')}")
        print(f"   Bitrate: {result.get('metrics', {}).get('bitrate_mbps', 0):.3f} Mbps")
        print(f"   PSNR: {result.get('metrics', {}).get('psnr_db', 0):.2f} dB")
        print(f"   SSIM: {result.get('metrics', {}).get('ssim', 0):.3f}")
        print(f"   Video: {result.get('metrics', {}).get('video_url', 'N/A')}")
        print(f"   Decoder: {result.get('metrics', {}).get('decoder_s3_key', 'N/A')}")
        return result
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

if __name__ == "__main__":
    print("🧪 Running controlled test experiments...")
    
    # Test 1: JPEG Quality 50
    result1 = send_experiment(
        f"controlled_test_q50_{int(time.time())}",
        test1_encoding,
        test1_decoding
    )
    
    time.sleep(5)
    
    # Test 2: JPEG Quality 80
    result2 = send_experiment(
        f"controlled_test_q80_{int(time.time())}",
        test2_encoding,
        test2_decoding
    )
    
    print("\n✅ All controlled tests sent!")

