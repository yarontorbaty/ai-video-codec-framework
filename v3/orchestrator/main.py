"""
V3.0 Orchestrator - Main Service

Coordinates video compression experiments using LLM-generated code.
Manages experiment lifecycle and evolution.
"""

import time
import logging
import os
import json
from typing import Dict, Optional

from llm_client import ClaudeClient
from experiment_manager import ExperimentManager
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Orchestrator:
    """Main orchestration service"""
    
    def __init__(self):
        self.config = Config()
        self.llm = ClaudeClient(self.config.anthropic_api_key)
        self.experiment_manager = ExperimentManager(
            self.config.worker_url,
            self.config.dynamodb_table
        )
        self.iteration = 0
    
    def run(self):
        """Main orchestration loop"""
        logger.info("="*80)
        logger.info("🚀 AI Video Codec Orchestrator v3.0")
        logger.info("="*80)
        logger.info(f"   Worker URL: {self.config.worker_url}")
        logger.info(f"   DynamoDB Table: {self.config.dynamodb_table}")
        logger.info(f"   Max Iterations: {self.config.max_iterations}")
        logger.info("="*80)
        
        try:
            while self.iteration < self.config.max_iterations:
                self.iteration += 1
                logger.info(f"\n{'='*80}")
                logger.info(f"🔄 ITERATION {self.iteration}")
                logger.info(f"{'='*80}\n")
                
                # Generate compression code
                logger.info("🤖 Generating compression code with LLM...")
                code = self.llm.generate_compression_code(
                    iteration=self.iteration,
                    previous_results=self.experiment_manager.get_recent_results(limit=5)
                )
                
                if not code:
                    logger.error("❌ Failed to generate code, skipping iteration")
                    time.sleep(60)
                    continue
                
                logger.info(f"✅ Generated {len(code['encoding'])} bytes encoding, {len(code['decoding'])} bytes decoding")
                
                # Run experiment
                logger.info("🎯 Submitting experiment to worker...")
                experiment_id = f"exp_iter{self.iteration}_{int(time.time())}"
                
                result = self.experiment_manager.run_experiment(
                    experiment_id=experiment_id,
                    encoding_code=code['encoding'],
                    decoding_code=code['decoding'],
                    iteration=self.iteration,
                    llm_reasoning=code.get('reasoning', '')
                )
                
                if result['status'] == 'success':
                    logger.info(f"✅ Experiment succeeded!")
                    logger.info(f"   PSNR: {result['metrics']['psnr_db']:.2f} dB")
                    logger.info(f"   SSIM: {result['metrics']['ssim']:.3f}")
                    logger.info(f"   Bitrate: {result['metrics']['bitrate_mbps']:.2f} Mbps")
                    logger.info(f"   Compression: {result['metrics']['compression_ratio']:.1f}x")
                else:
                    logger.error(f"❌ Experiment failed: {result.get('error')}")
                
                # Wait before next iteration
                logger.info(f"\n⏳ Waiting {self.config.iteration_delay_sec}s before next iteration...")
                time.sleep(self.config.iteration_delay_sec)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Orchestrator stopped by user")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}", exc_info=True)
        
        logger.info(f"\n✅ Orchestrator completed {self.iteration} iterations")


def main():
    """Entry point"""
    orchestrator = Orchestrator()
    orchestrator.run()


if __name__ == '__main__':
    main()

