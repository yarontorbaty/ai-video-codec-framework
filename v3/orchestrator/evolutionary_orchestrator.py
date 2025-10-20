"""
V3.0 EVOLUTIONARY Orchestrator - Learns from best performers!

This version implements an evolutionary algorithm:
1. Query top performers from previous generation
2. Feed them to Claude with "improve upon these" prompt
3. Test the new generation
4. Repeat - systematic improvement!
"""

import anthropic
import boto3
import json
import time
import logging
import requests
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
secrets_client = boto3.client('secretsmanager', region_name='us-east-1')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
experiments_table = dynamodb.Table('ai-codec-v3-fast-experiments')

# Get API key
secret = secrets_client.get_secret_value(SecretId='ai-video-codec/anthropic-api-key')
ANTHROPIC_API_KEY = json.loads(secret['SecretString'])['ANTHROPIC_API_KEY']

# Initialize Claude client
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Configuration
WORKER_URL = os.environ.get('WORKER_URL', 'http://localhost:8080')
BATCH_SIZE = 20
GENERATION_SIZE = 100  # Experiments per generation

# Base system prompt for exploration
BASE_SYSTEM_PROMPT = """You are a video compression codec generator. Generate SIMPLE, FAST compression algorithms.

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

def run_encoding_agent(frames):
    # frames: list of numpy arrays (64, 64, 3) uint8
    # return: bytes (compressed data)
    return compressed_bytes

DECODING FUNCTION TEMPLATE:
import numpy as np
import cv2
import pickle

def run_decoding_agent(compressed_data, expected_frames):
    # compressed_data: bytes
    # expected_frames: int (should be 10)
    # return: list of numpy arrays (64, 64, 3) uint8
    return frames

COMPRESSION IDEAS:
- Downsampling, DCT/DWT, Quantization, Frame differencing
- Run-length encoding, Motion compensation, Color space transforms

Generate 10 DIFFERENT codec variations.
Output as JSON array: [{"encoding_code": "...", "decoding_code": "..."}, ...]"""


class EvolutionaryOrchestrator:
    """Orchestrator with evolutionary learning"""
    
    def __init__(self, worker_url: str):
        self.worker_url = worker_url
        self.generation = 0
        logger.info(f"🧬 Evolutionary Orchestrator initialized")
        logger.info(f"Worker URL: {worker_url}")
    
    def run_evolutionary(self, num_generations: int = 100):
        """Run evolutionary experiment loop"""
        start_time = time.time()
        total_experiments = 0
        
        logger.info("=" * 80)
        logger.info(f"🧬 EVOLUTIONARY MODE - {num_generations} generations")
        logger.info("=" * 80)
        
        for gen in range(num_generations):
            self.generation = gen
            gen_start = time.time()
            
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"🧬 GENERATION {gen + 1}/{num_generations}")
            logger.info("=" * 80)
            
            # Get best performers from previous generation
            if gen == 0:
                logger.info("📊 Generation 0: Random exploration (no prior data)")
                best_performers = []
            else:
                logger.info("📊 Querying top performers from previous generation...")
                best_performers = self._get_top_performers(limit=5)
                if best_performers:
                    logger.info(f"✅ Found {len(best_performers)} top performers:")
                    for i, perf in enumerate(best_performers, 1):
                        logger.info(f"   {i}. Compression: {perf['compression_ratio']:.2f}x | MSE: {perf['mse']:.2f}")
                else:
                    logger.info("⚠️  No successful experiments yet, using random exploration")
            
            # Generate new codecs (with or without context)
            logger.info(f"🤖 Generating {GENERATION_SIZE} codecs with 10 parallel Claude calls...")
            all_codecs = self._generate_codecs_with_feedback(best_performers, num_calls=10)
            logger.info(f"✅ Generated {len(all_codecs)} codecs")
            
            # Test all codecs in this generation
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
                    time.sleep(5)
            
            # Generation summary
            gen_time = time.time() - gen_start
            logger.info("")
            logger.info(f"📊 Generation {gen} Summary:")
            logger.info(f"   Experiments: {gen_experiments}")
            logger.info(f"   Time: {gen_time:.1f}s")
            logger.info(f"   Rate: {gen_experiments/gen_time:.1f} exp/sec")
            
            # Check for improvement
            if gen > 0:
                current_best = self._get_top_performers(limit=1)
                if current_best:
                    logger.info(f"   🏆 Current Best: {current_best[0]['compression_ratio']:.2f}x compression, {current_best[0]['mse']:.2f} MSE")
        
        total_time = time.time() - start_time
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"🎉 EVOLUTIONARY RUN COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Generations: {num_generations}")
        logger.info(f"Total Experiments: {total_experiments}")
        logger.info(f"Total Time: {total_time/60:.1f} minutes")
        logger.info(f"Average Rate: {total_experiments/total_time:.1f} exp/sec")
        
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
            # Scan for successful experiments
            response = experiments_table.scan(
                FilterExpression='#s = :status',
                ExpressionAttributeNames={'#s': 'status'},
                ExpressionAttributeValues={':status': 'success'},
                Limit=1000  # Get recent successful ones
            )
            
            experiments = response['Items']
            
            # Convert Decimals to floats and flatten metrics
            for exp in experiments:
                metrics = exp.get('metrics', {})
                # Handle both old flat schema and new nested metrics schema
                if 'compression_ratio' in exp:
                    exp['compression_ratio'] = float(exp['compression_ratio'])
                elif 'compression_ratio' in metrics:
                    exp['compression_ratio'] = float(metrics['compression_ratio'])
                else:
                    exp['compression_ratio'] = 0
                
                if 'mse' in exp:
                    exp['mse'] = float(exp['mse'])
                elif 'mse' in metrics:
                    exp['mse'] = float(metrics['mse'])
                else:
                    exp['mse'] = 9999
                
                if 'generation' in exp:
                    exp['generation'] = int(exp['generation'])
            
            # Sort by compression ratio (higher is better)
            experiments.sort(key=lambda x: x.get('compression_ratio', 0), reverse=True)
            
            # Return top performers
            return experiments[:limit]
            
        except Exception as e:
            logger.error(f"❌ Failed to query top performers: {e}")
            return []
    
    def _generate_codecs_with_feedback(self, best_performers: List[Dict], num_calls: int = 10) -> List[Dict]:
        """Generate codecs with evolutionary feedback"""
        
        # Build context-aware prompt
        if best_performers:
            # EVOLUTIONARY MODE: Improve upon best
            context = "PREVIOUS TOP PERFORMERS:\n\n"
            for i, perf in enumerate(best_performers, 1):
                context += f"Performer {i}: {perf['compression_ratio']:.2f}x compression, {perf['mse']:.2f} MSE\n"
            
            system_prompt = f"""{BASE_SYSTEM_PROMPT}

{context}

YOUR TASK: Generate 10 NEW codecs that IMPROVE upon these top performers.
- Try variations of their techniques
- Combine their best features
- Fix their weaknesses (if high MSE, improve quality)
- Try to beat the compression ratio while maintaining quality
- Be creative but build upon what worked!"""
        else:
            # EXPLORATION MODE: Random search
            system_prompt = BASE_SYSTEM_PROMPT
        
        def call_claude_sync(call_idx):
            """Single Claude API call"""
            try:
                result = self._generate_codec_batch(system_prompt)
                logger.info(f"✅ Call {call_idx+1}/{num_calls} complete: {len(result)} codecs")
                return result
            except Exception as e:
                logger.error(f"❌ Call {call_idx+1} failed: {e}")
                return []
        
        # Parallel Claude calls
        all_codecs = []
        with ThreadPoolExecutor(max_workers=num_calls) as executor:
            futures = [executor.submit(call_claude_sync, i) for i in range(num_calls)]
            for future in futures:
                codecs = future.result()
                all_codecs.extend(codecs)
        
        return all_codecs
    
    def _generate_codec_batch(self, system_prompt: str) -> List[Dict]:
        """Generate a batch using Claude"""
        try:
            message = claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                temperature=1.0,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": f"Generate 10 different fast video compression codecs for 64x64 videos. Generation {self.generation}."
                }]
            )
            
            content = message.content[0].text
            
            # Robust JSON extraction
            codecs_raw = self._extract_json_robust(content)
            if not codecs_raw:
                return []
            
            # Normalize codec format
            codecs = []
            for codec in codecs_raw:
                encoding = (codec.get('encoding_code') or codec.get('encode') or 
                           codec.get('encoder') or codec.get('encoding_function'))
                decoding = (codec.get('decoding_code') or codec.get('decode') or 
                           codec.get('decoder') or codec.get('decoding_function'))
                
                if encoding and decoding:
                    codecs.append({
                        'encoding_code': encoding,
                        'decoding_code': decoding
                    })
            
            return codecs
            
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return []
    
    def _extract_json_robust(self, content: str) -> List[Dict]:
        """Robustly extract JSON from Claude's response, handling markdown and formatting issues"""
        import re
        
        # Strategy 1: Strip markdown code blocks
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        # Strategy 2: Find JSON array
        start = content.find('[')
        end = content.rfind(']') + 1
        
        if start == -1 or end == 0:
            logger.warning("No JSON array found in response")
            return []
        
        json_str = content[start:end]
        
        # Strategy 3: Try parsing as-is
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial parse failed: {e}")
        
        # Strategy 4: Fix common issues
        try:
            # Replace literal newlines in strings with \n
            json_str = re.sub(r'(?<!\\)\n(?=\s*")', r'\\n', json_str)
            
            # Fix unescaped quotes in strings (heuristic)
            json_str = json_str.replace('\\"', '"')  # Unescape all first
            json_str = re.sub(r'(?<!\\)"(?=\s*[^,\]\}:\[])', r'\\"', json_str)  # Re-escape where needed
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Repair attempt failed: {e}")
        
        # Strategy 5: Extract individual objects
        try:
            # Find all object patterns
            objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_str)
            codecs = []
            for obj_str in objects:
                try:
                    codec = json.loads(obj_str)
                    if 'encoding_code' in codec or 'encode' in codec:
                        codecs.append(codec)
                except:
                    continue
            
            if codecs:
                logger.info(f"✅ Recovered {len(codecs)} codecs via object extraction")
                return codecs
        except Exception as e:
            logger.warning(f"Object extraction failed: {e}")
        
        # Strategy 6: Last resort - try to extract code blocks manually
        try:
            encoding_blocks = re.findall(r'"encoding_code":\s*"([^"]+)"', content, re.DOTALL)
            decoding_blocks = re.findall(r'"decoding_code":\s*"([^"]+)"', content, re.DOTALL)
            
            if len(encoding_blocks) == len(decoding_blocks):
                codecs = []
                for enc, dec in zip(encoding_blocks, decoding_blocks):
                    codecs.append({
                        'encoding_code': enc.replace('\\n', '\n').replace('\\"', '"'),
                        'decoding_code': dec.replace('\\n', '\n').replace('\\"', '"')
                    })
                if codecs:
                    logger.info(f"✅ Recovered {len(codecs)} codecs via regex extraction")
                    return codecs
        except Exception as e:
            logger.warning(f"Regex extraction failed: {e}")
        
        logger.error("❌ All JSON extraction strategies failed")
        return []


def main():
    import sys
    
    worker_url = sys.argv[1] if len(sys.argv) > 1 else WORKER_URL
    num_generations = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    orchestrator = EvolutionaryOrchestrator(worker_url)
    logger.info(f"🎯 Target: {num_generations} generations, ~{num_generations * GENERATION_SIZE} total experiments")
    orchestrator.run_evolutionary(num_generations=num_generations)


if __name__ == "__main__":
    main()

