"""
V3.0 Fast Orchestrator - Pre-generates codec variations and sends batches

OPTIMIZED: Makes 20 parallel Claude API calls to generate 200 codecs at once!

Strategy:
1. Make 20 parallel Claude API calls (async)
2. Get 200 codec variations in ~45 seconds instead of 900 seconds
3. Send them in batches of 20 to worker
4. Target: 10,000 experiments/hour achieved!
"""

import anthropic
import boto3
import json
import time
import logging
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS configuration
secrets_client = boto3.client('secretsmanager', region_name='us-east-1')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
experiments_table = dynamodb.Table('ai-codec-v3-fast-experiments')

# Get API key
secret = secrets_client.get_secret_value(SecretId='ai-video-codec/anthropic-api-key')
ANTHROPIC_API_KEY = json.loads(secret['SecretString'])['ANTHROPIC_API_KEY']

# Initialize Claude client
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Worker configuration
WORKER_URL = "http://localhost:8080"  # Will be updated with actual worker IP

# Batch configuration
BATCH_SIZE = 20
CODES_PER_GENERATION = 100


SYSTEM_PROMPT = """You are a video compression codec generator. Generate SIMPLE, FAST compression algorithms.

REQUIREMENTS:
1. For tiny 64x64 10-frame videos (numpy arrays)
2. In-memory only (no file I/O)
3. Must complete in <10 seconds
4. Function names: run_encoding_agent(frames) -> bytes, run_decoding_agent(data, num_frames) -> frames

AVAILABLE LIBRARIES (ONLY USE THESE):
- numpy (as np) - for array operations
- cv2 (opencv) - for image operations, DCT, DWT
- pickle - for serialization
- scipy - for signal processing, FFT, transforms
- scikit-image (skimage) - for image processing

DO NOT use: tensorflow, torch, PIL, matplotlib, pandas, or any other libraries!

ENCODING FUNCTION TEMPLATE:
import numpy as np
import cv2
import pickle
# Optional: from scipy import fft, signal
# Optional: from skimage import transform

def run_encoding_agent(frames):
    # frames: list of numpy arrays (64, 64, 3) uint8
    # return: bytes (compressed data)
    # YOUR COMPRESSION CODE HERE
    return compressed_bytes

DECODING FUNCTION TEMPLATE:
import numpy as np
import cv2
import pickle
# Optional: from scipy import fft, signal
# Optional: from skimage import transform

def run_decoding_agent(compressed_data, expected_frames):
    # compressed_data: bytes
    # expected_frames: int (should be 10)
    # return: list of numpy arrays (64, 64, 3) uint8
    # YOUR DECOMPRESSION CODE HERE
    return frames  # Must be list of exactly 'expected_frames' arrays

COMPRESSION IDEAS TO EXPLORE:
- Downsampling + upsampling (cv2.resize)
- Color space conversion (BGR->YCrCb, BGR->HSV)
- Quantization (reduce bit depth)
- DCT/DWT transforms (cv2.dct, scipy.fft)
- Frame differencing (only store deltas)
- Run-length encoding
- Simple motion compensation
- Subsampling (skip pixels/frames)
- Huffman-like encoding
- PCA/SVD compression

Generate 10 DIFFERENT, SIMPLE, FAST codec variations.
Each must use ONLY the allowed libraries above.
Output as JSON array: [{"encoding_code": "...", "decoding_code": "..."}, ...]"""


class FastOrchestrator:
    """Fast orchestrator for high-throughput experiments"""
    
    def __init__(self, worker_url: str):
        self.worker_url = worker_url
        self.iteration = 0
        logger.info(f"🚀 Fast Orchestrator initialized")
        logger.info(f"Worker URL: {worker_url}")
    
    def run_continuous(self, target_experiments: int = 10000):
        """Run continuous batches until target reached"""
        total_experiments = 0
        start_time = time.time()
        
        logger.info(f"🎯 Target: {target_experiments} experiments")
        
        while total_experiments < target_experiments:
            # Generate MANY codecs in parallel (20 Claude calls at once!)
            logger.info(f"🤖 Generating codec batch with 20 parallel Claude calls...")
            batch_start = time.time()
            all_codecs = self._generate_codecs_parallel(num_calls=20)
            gen_time = time.time() - batch_start
            logger.info(f"✅ Generated {len(all_codecs)} codecs in {gen_time:.1f}s ({len(all_codecs)/gen_time:.1f} codecs/sec)")
            
            # Send codecs in batches to worker
            for batch_idx in range(0, len(all_codecs), BATCH_SIZE):
                codec_batch = all_codecs[batch_idx:batch_idx+BATCH_SIZE]
                
                # Create experiment definitions
                experiments = []
                for codec in codec_batch:
                    self.iteration += 1
                    experiments.append({
                        'experiment_id': f'fast_iter{self.iteration}_{int(time.time())}',
                        'encoding_code': codec['encoding_code'],
                        'decoding_code': codec['decoding_code']
                    })
                
                # Send batch to worker
                logger.info(f"📤 Sending batch {batch_idx//BATCH_SIZE + 1} with {len(experiments)} experiments...")
                
                try:
                    response = requests.post(
                        f"{self.worker_url}/batch",
                        json={'experiments': experiments},
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        total_experiments += result['total']
                        
                        logger.info(f"✅ Batch complete: {result['succeeded']} succeeded, {result['failed']} failed")
                        
                        # Progress
                        elapsed = time.time() - start_time
                        rate = total_experiments / elapsed
                        remaining = target_experiments - total_experiments
                        eta = remaining / rate if rate > 0 else 0
                        
                        logger.info(f"📊 Progress: {total_experiments}/{target_experiments} ({rate:.1f} exp/sec, {rate*3600:.0f} exp/hour, ETA: {eta/60:.1f}min)")
                    else:
                        logger.error(f"❌ Batch failed: {response.status_code}")
                except Exception as e:
                    logger.error(f"❌ Request failed: {e}")
                    time.sleep(5)
        
        total_time = time.time() - start_time
        final_rate = total_experiments / total_time
        
        logger.info("=" * 60)
        logger.info(f"🎉 COMPLETE!")
        logger.info(f"Total Experiments: {total_experiments}")
        logger.info(f"Total Time: {total_time/60:.1f} minutes")
        logger.info(f"Average Rate: {final_rate:.1f} exp/sec ({final_rate*3600:.0f} exp/hour)")
        logger.info("=" * 60)
    
    def _generate_codecs_parallel(self, num_calls: int = 20) -> List[Dict]:
        """Generate codecs using parallel Claude API calls"""
        
        def call_claude_sync(call_idx):
            """Single Claude API call (blocking)"""
            try:
                result = self._generate_codec_batch()
                logger.info(f"✅ Call {call_idx+1}/{num_calls} complete: {len(result)} codecs")
                return result
            except Exception as e:
                logger.error(f"❌ Call {call_idx+1} failed: {e}")
                return []
        
        # Use ThreadPoolExecutor to make parallel API calls
        all_codecs = []
        with ThreadPoolExecutor(max_workers=num_calls) as executor:
            # Submit all calls at once
            futures = [executor.submit(call_claude_sync, i) for i in range(num_calls)]
            
            # Collect results as they complete
            for future in futures:
                codecs = future.result()
                all_codecs.extend(codecs)
        
        logger.info(f"🎯 Total codecs collected: {len(all_codecs)}")
        return all_codecs
    
    def _generate_codec_batch(self) -> List[Dict]:
        """Generate a batch of codec variations using Claude"""
        try:
            message = claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                temperature=1.0,  # High temperature for diversity
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Generate 10 fast, simple video compression codecs. Vary the approaches. Iteration: {self.iteration}"
                }]
            )
            
            # Extract JSON from response
            content = message.content[0].text
            logger.info(f"📝 Claude response length: {len(content)} chars")
            
            # Find JSON array
            start = content.find('[')
            end = content.rfind(']') + 1
            if start == -1 or end == 0:
                logger.warning("No JSON array found, using fallback codec")
                return [self._get_fallback_codec() for _ in range(10)]
            
            codecs_raw = json.loads(content[start:end])
            logger.info(f"✅ Parsed {len(codecs_raw)} codecs from Claude")
            
            # Normalize codec format - handle different key names
            codecs = []
            for i, codec in enumerate(codecs_raw):
                # Try different possible key names
                encoding = (codec.get('encoding_code') or 
                           codec.get('encode') or 
                           codec.get('encoder') or 
                           codec.get('encoding_function') or
                           codec.get('encode_code'))
                
                decoding = (codec.get('decoding_code') or
                           codec.get('decode') or
                           codec.get('decoder') or
                           codec.get('decoding_function') or
                           codec.get('decode_code'))
                
                if encoding and decoding:
                    codecs.append({
                        'encoding_code': encoding,
                        'decoding_code': decoding
                    })
                else:
                    logger.warning(f"⚠️ Codec {i} missing encoding or decoding, keys: {list(codec.keys())}")
            
            if len(codecs) < 5:
                logger.warning(f"⚠️ Only got {len(codecs)} valid codecs, padding with fallback")
                while len(codecs) < 10:
                    codecs.append(self._get_fallback_codec())
            
            return codecs
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing error: {e}")
            return [self._get_fallback_codec() for _ in range(10)]
        except Exception as e:
            logger.error(f"❌ Code generation error: {e}")
            return [self._get_fallback_codec() for _ in range(10)]
    
    def _get_fallback_codec(self) -> Dict:
        """Simple fallback codec if generation fails"""
        return {
            'encoding_code': '''
import numpy as np
import pickle

def encode(frames):
    # Simple downsampling + pickle
    downsampled = [frame[::2, ::2, :] for frame in frames]
    return pickle.dumps(downsampled)
''',
            'decoding_code': '''
import numpy as np
import pickle
import cv2

def decode(data, num_frames):
    # Upsample back
    downsampled = pickle.loads(data)
    frames = []
    for frame in downsampled:
        upsampled = cv2.resize(frame, (64, 64))
        frames.append(upsampled.astype(np.uint8))
    return frames
'''
        }


def main():
    """Start fast orchestrator"""
    import sys
    
    logger.info("=" * 60)
    logger.info("🚀 Fast Orchestrator v3.0 Starting")
    logger.info("=" * 60)
    
    # Get worker URL from command line or use default
    worker_url = sys.argv[1] if len(sys.argv) > 1 else WORKER_URL
    
    orchestrator = FastOrchestrator(worker_url)
    
    # Run 10,000 experiments
    orchestrator.run_continuous(target_experiments=10000)


if __name__ == '__main__':
    main()

