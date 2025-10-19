"""
V3.0 LOCAL LLM Evolutionary Orchestrator - No rate limits!

Uses a local vLLM server instead of Claude for unlimited codec generation.
"""

import requests
import boto3
import json
import time
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from decimal import Decimal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
experiments_table = dynamodb.Table('ai-codec-v3-fast-experiments')

# Configuration
WORKER_URL = os.environ.get('WORKER_URL', 'http://localhost:8080')
LOCAL_LLM_URL = os.environ.get('LOCAL_LLM_URL', 'http://localhost:8000/v1/chat/completions')
BATCH_SIZE = 20
GENERATION_SIZE = 100

# System prompt
BASE_SYSTEM_PROMPT = """You are a video compression codec generator. Generate SIMPLE, FAST compression algorithms.

REQUIREMENTS:
1. For tiny 64x64 10-frame videos (numpy arrays)
2. In-memory only (no file I/O)
3. Must complete in <10 seconds
4. Function names: run_encoding_agent(frames) -> bytes, run_decoding_agent(data, num_frames) -> frames

AVAILABLE LIBRARIES:
- numpy (as np), cv2 (opencv), pickle, scipy, scikit-image (skimage)

ENCODING FUNCTION:
import numpy as np
import cv2
import pickle

def run_encoding_agent(frames):
    # frames: list of numpy arrays (64, 64, 3) uint8
    return compressed_bytes

DECODING FUNCTION:
import numpy as np
import cv2
import pickle

def run_decoding_agent(compressed_data, expected_frames):
    # compressed_data: bytes, expected_frames: int
    return frames  # list of numpy arrays

Generate 10 DIFFERENT codec variations.
Output ONLY valid JSON array: [{"encoding_code": "...", "decoding_code": "..."}, ...]
No markdown, no explanation, just the JSON array."""


class LocalLLMOrchestrator:
    """Orchestrator using local LLM"""
    
    def __init__(self, worker_url: str, llm_url: str):
        self.worker_url = worker_url
        self.llm_url = llm_url
        self.generation = 0
        logger.info(f"🧬 Local LLM Orchestrator initialized")
        logger.info(f"Worker URL: {worker_url}")
        logger.info(f"LLM URL: {llm_url}")
    
    def run_evolutionary(self, num_generations: int = 100):
        """Run evolutionary loop with local LLM"""
        start_time = time.time()
        total_experiments = 0
        
        logger.info("=" * 80)
        logger.info(f"🧬 LOCAL LLM EVOLUTIONARY MODE - {num_generations} generations")
        logger.info(f"🚀 NO RATE LIMITS - UNLIMITED SPEED!")
        logger.info("=" * 80)
        
        for gen in range(num_generations):
            self.generation = gen
            gen_start = time.time()
            
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"🧬 GENERATION {gen + 1}/{num_generations}")
            logger.info("=" * 80)
            
            # Get best performers
            if gen == 0:
                logger.info("📊 Generation 0: Random exploration")
                best_performers = []
            else:
                logger.info("📊 Querying top performers...")
                best_performers = self._get_top_performers(limit=5)
                if best_performers:
                    logger.info(f"✅ Top 5 performers:")
                    for i, perf in enumerate(best_performers, 1):
                        logger.info(f"   {i}. {perf['compression_ratio']:.2f}x | MSE: {perf['mse']:.2f}")
            
            # Generate codecs with local LLM (FAST!)
            logger.info(f"🤖 Generating {GENERATION_SIZE} codecs with LOCAL LLM (10 parallel calls)...")
            all_codecs = self._generate_codecs_parallel(best_performers, num_calls=10)
            logger.info(f"✅ Generated {len(all_codecs)} codecs")
            
            # Test all codecs
            gen_experiments = 0
            for batch_idx in range(0, len(all_codecs), BATCH_SIZE):
                codec_batch = all_codecs[batch_idx:batch_idx+BATCH_SIZE]
                
                experiments = [{
                    'experiment_id': f'gen{gen}_exp{gen_experiments + i}_{int(time.time())}',
                    'generation': gen,
                    'encoding_code': codec['encoding_code'],
                    'decoding_code': codec['decoding_code']
                } for i, codec in enumerate(codec_batch)]
                
                try:
                    response = requests.post(
                        f"{self.worker_url}/batch",
                        json={'experiments': experiments, 'generation': gen},
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        gen_experiments += result['total']
                        total_experiments += result['total']
                        logger.info(f"✅ Batch {batch_idx//BATCH_SIZE + 1}: {result['succeeded']} succeeded, {result['failed']} failed")
                    else:
                        logger.error(f"❌ Batch failed: {response.status_code}")
                except Exception as e:
                    logger.error(f"❌ Request failed: {e}")
            
            # Generation summary
            gen_time = time.time() - gen_start
            rate = gen_experiments / gen_time if gen_time > 0 else 0
            logger.info("")
            logger.info(f"📊 Generation {gen} Summary:")
            logger.info(f"   Experiments: {gen_experiments}")
            logger.info(f"   Time: {gen_time:.1f}s")
            logger.info(f"   Rate: {rate:.1f} exp/sec")
            
            # Check improvement
            if gen > 0 and best_performers:
                current_best = self._get_top_performers(limit=1)
                if current_best:
                    improvement = current_best[0]['compression_ratio'] - best_performers[0]['compression_ratio']
                    logger.info(f"   🏆 Best: {current_best[0]['compression_ratio']:.2f}x (Δ {improvement:+.2f}x)")
        
        total_time = time.time() - start_time
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"🎉 LOCAL LLM EVOLUTIONARY RUN COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Generations: {num_generations}")
        logger.info(f"Total Experiments: {total_experiments}")
        logger.info(f"Total Time: {total_time/60:.1f} minutes")
        logger.info(f"Average Rate: {total_experiments/total_time:.1f} exp/sec ({total_experiments/total_time*3600:.0f} exp/hour)")
        
        # Final top performers
        final_best = self._get_top_performers(limit=10)
        if final_best:
            logger.info("")
            logger.info("🏆 TOP 10 FINAL PERFORMERS:")
            for i, perf in enumerate(final_best, 1):
                logger.info(f"   {i}. Gen {perf.get('generation', '?')}: {perf['compression_ratio']:.2f}x | MSE: {perf['mse']:.2f}")
        
        logger.info("=" * 80)
    
    def _get_top_performers(self, limit: int = 5) -> List[Dict]:
        """Query top performers from DynamoDB"""
        try:
            response = experiments_table.scan(
                FilterExpression='#s = :status',
                ExpressionAttributeNames={'#s': 'status'},
                ExpressionAttributeValues={':status': 'success'},
                Limit=1000
            )
            
            experiments = response['Items']
            
            for exp in experiments:
                if 'compression_ratio' in exp:
                    exp['compression_ratio'] = float(exp['compression_ratio'])
                if 'mse' in exp:
                    exp['mse'] = float(exp['mse'])
                if 'generation' in exp:
                    exp['generation'] = int(exp['generation'])
            
            experiments.sort(key=lambda x: x.get('compression_ratio', 0), reverse=True)
            return experiments[:limit]
            
        except Exception as e:
            logger.error(f"❌ Failed to query: {e}")
            return []
    
    def _generate_codecs_parallel(self, best_performers: List[Dict], num_calls: int = 10) -> List[Dict]:
        """Generate codecs with local LLM (parallel)"""
        
        # Build prompt
        if best_performers:
            context = "\n\nPREVIOUS TOP PERFORMERS:\n"
            for i, perf in enumerate(best_performers, 1):
                context += f"{i}. {perf['compression_ratio']:.2f}x compression, {perf['mse']:.2f} MSE\n"
            prompt = f"{BASE_SYSTEM_PROMPT}{context}\n\nYOUR TASK: Generate 10 NEW codecs that IMPROVE upon these. Try variations and combinations."
        else:
            prompt = BASE_SYSTEM_PROMPT
        
        def call_llm_sync(call_idx):
            """Single LLM call"""
            try:
                result = self._generate_codec_batch(prompt)
                logger.info(f"✅ Call {call_idx+1}/{num_calls}: {len(result)} codecs")
                return result
            except Exception as e:
                logger.error(f"❌ Call {call_idx+1} failed: {e}")
                return []
        
        # Parallel calls
        all_codecs = []
        with ThreadPoolExecutor(max_workers=num_calls) as executor:
            futures = [executor.submit(call_llm_sync, i) for i in range(num_calls)]
            for future in futures:
                codecs = future.result()
                all_codecs.extend(codecs)
        
        return all_codecs
    
    def _generate_codec_batch(self, prompt: str) -> List[Dict]:
        """Generate batch using local LLM"""
        try:
            # Call vLLM server
            response = requests.post(
                self.llm_url,
                json={
                    "model": "llama-3.1-8b",
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Generate 10 different video compression codecs. Generation {self.generation}. Output ONLY the JSON array."}
                    ],
                    "max_tokens": 4000,
                    "temperature": 1.0
                },
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"LLM error: {response.status_code}")
                return []
            
            content = response.json()['choices'][0]['message']['content']
            
            # Extract JSON
            start = content.find('[')
            end = content.rfind(']') + 1
            if start == -1 or end == 0:
                logger.warning("No JSON found")
                return []
            
            try:
                codecs_raw = json.loads(content[start:end])
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                return []
            
            # Normalize
            codecs = []
            for codec in codecs_raw:
                encoding = (codec.get('encoding_code') or codec.get('encode'))
                decoding = (codec.get('decoding_code') or codec.get('decode'))
                
                if encoding and decoding:
                    codecs.append({
                        'encoding_code': encoding,
                        'decoding_code': decoding
                    })
            
            return codecs
            
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return []


def main():
    import sys
    
    worker_url = sys.argv[1] if len(sys.argv) > 1 else WORKER_URL
    llm_url = sys.argv[2] if len(sys.argv) > 2 else LOCAL_LLM_URL
    num_generations = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    orchestrator = LocalLLMOrchestrator(worker_url, llm_url)
    logger.info(f"🎯 Target: {num_generations} generations, ~{num_generations * GENERATION_SIZE} total experiments")
    logger.info(f"💰 Cost: ~${num_generations * 0.5 / 60:.2f} (GPU time only)")
    orchestrator.run_evolutionary(num_generations=num_generations)


if __name__ == "__main__":
    main()

