"""
V3.0 Fast Orchestrator - Pre-generates codec variations and sends batches

Strategy:
1. Pre-generate 100 codec variations
2. Send them in batches of 20 to worker
3. Analyze results and generate next batch
4. Target: 10,000 experiments/hour
"""

import anthropic
import boto3
import json
import time
import logging
import requests
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
secret = secrets_client.get_secret_value(SecretId='anthropic-api-key')
ANTHROPIC_API_KEY = json.loads(secret['SecretString'])['api_key']

# Initialize Claude client
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Worker configuration
WORKER_URL = "http://localhost:8080"  # Will be updated with actual worker IP

# Batch configuration
BATCH_SIZE = 20
CODES_PER_GENERATION = 100


SYSTEM_PROMPT = """You are a video compression codec generator. Generate SIMPLE, FAST compression algorithms.

REQUIREMENTS:
1. For tiny 64x64 10-frame videos
2. In-memory only (no file I/O)
3. Must complete in <10 seconds
4. encode(frames) -> bytes
5. decode(data, num_frames) -> frames

ENCODING FUNCTION:
def encode(frames):
    # frames: list of numpy arrays (H,W,3)
    # return: bytes (compressed data)
    import numpy as np
    import pickle
    # YOUR CODE HERE
    return compressed_bytes

DECODING FUNCTION:
def decode(data, num_frames):
    # data: bytes (compressed)
    # num_frames: int (expected number of frames)
    # return: list of numpy arrays (H,W,3) uint8
    import numpy as np
    import pickle
    # YOUR CODE HERE
    return frames

Generate 10 different compression approaches. Focus on SPEED and SIMPLICITY.
Ideas: downsampling, color quantization, frame differencing, DCT, simple predictive coding.

Output JSON array with 10 codec variations."""


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
            # Generate batch of codecs
            logger.info(f"🤖 Generating codec batch...")
            codecs = self._generate_codec_batch()
            
            # Create experiment definitions
            experiments = []
            for i, codec in enumerate(codecs):
                self.iteration += 1
                experiments.append({
                    'experiment_id': f'fast_iter{self.iteration}_{int(time.time())}',
                    'encoding_code': codec['encoding_code'],
                    'decoding_code': codec['decoding_code']
                })
            
            # Send batch to worker
            logger.info(f"📤 Sending batch of {len(experiments)} experiments...")
            batch_start = time.time()
            
            response = requests.post(
                f"{self.worker_url}/batch",
                json={'experiments': experiments},
                timeout=300  # 5 minute timeout for batch
            )
            
            batch_time = time.time() - batch_start
            
            if response.status_code == 200:
                result = response.json()
                total_experiments += result['total']
                
                logger.info(f"✅ Batch complete:")
                logger.info(f"   Total: {result['total']}")
                logger.info(f"   Success: {result['succeeded']}")
                logger.info(f"   Failed: {result['failed']}")
                logger.info(f"   Time: {batch_time:.1f}s")
                logger.info(f"   Rate: {result['total']/batch_time:.1f} exp/sec")
                
                # Progress
                elapsed = time.time() - start_time
                rate = total_experiments / elapsed
                remaining = target_experiments - total_experiments
                eta = remaining / rate if rate > 0 else 0
                
                logger.info(f"📊 Progress: {total_experiments}/{target_experiments} ({rate:.1f} exp/sec, ETA: {eta/60:.1f}min)")
            else:
                logger.error(f"❌ Batch failed: {response.status_code}")
                time.sleep(5)
        
        total_time = time.time() - start_time
        final_rate = total_experiments / total_time
        
        logger.info("=" * 60)
        logger.info(f"🎉 COMPLETE!")
        logger.info(f"Total Experiments: {total_experiments}")
        logger.info(f"Total Time: {total_time/60:.1f} minutes")
        logger.info(f"Average Rate: {final_rate:.1f} exp/sec ({final_rate*3600:.0f} exp/hour)")
        logger.info("=" * 60)
    
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
            
            # Find JSON array
            start = content.find('[')
            end = content.rfind(']') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON array found in response")
            
            codecs = json.loads(content[start:end])
            
            if len(codecs) < 5:
                logger.warning(f"⚠️ Only got {len(codecs)} codecs, expected 10")
            
            return codecs
            
        except Exception as e:
            logger.error(f"❌ Code generation error: {e}")
            # Return fallback simple codec
            return [self._get_fallback_codec()]
    
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

