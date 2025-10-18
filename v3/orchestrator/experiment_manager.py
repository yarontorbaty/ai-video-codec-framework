"""
V3.0 Experiment Manager

Manages experiment lifecycle: submission, tracking, storage
"""

import requests
import boto3
import json
import time
import logging
from decimal import Decimal
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Manage experiments from submission to storage"""
    
    def __init__(self, worker_url: str, dynamodb_table: str):
        self.worker_url = worker_url
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        self.table = self.dynamodb.Table(dynamodb_table)
    
    def run_experiment(
        self,
        experiment_id: str,
        encoding_code: str,
        decoding_code: str,
        iteration: int,
        llm_reasoning: str = ""
    ) -> Dict:
        """
        Submit experiment to worker and store results
        
        Returns:
            {
                'status': 'success' | 'failed',
                'metrics': {...},
                'error': str (if failed)
            }
        """
        try:
            # Mark as in_progress in DynamoDB
            timestamp = int(time.time())
            self._store_in_progress(experiment_id, iteration, timestamp)
            
            # Prepare experiment payload
            payload = {
                'experiment_id': experiment_id,
                'encoding_code': encoding_code,
                'decoding_code': decoding_code,
                'timestamp': timestamp
            }
            
            # Submit to worker
            logger.info(f"📤 Sending experiment to worker: {self.worker_url}")
            response = requests.post(
                f"{self.worker_url}/experiment",
                json=payload,
                timeout=300  # 5 minutes max
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Store in DynamoDB
            self._store_result(
                experiment_id=experiment_id,
                result=result,
                iteration=iteration,
                llm_reasoning=llm_reasoning
            )
            
            experiment_result = result.get('result', {})
            return {
                'status': experiment_result.get('status', 'unknown'),
                'metrics': experiment_result.get('metrics', {}),
                'error': experiment_result.get('error')
            }
            
        except requests.exceptions.Timeout:
            logger.error(f"❌ Worker timeout after 5 minutes")
            self._store_failed(experiment_id, "Worker timeout", iteration)
            return {'status': 'failed', 'error': 'Worker timeout'}
            
        except Exception as e:
            logger.error(f"❌ Experiment error: {e}", exc_info=True)
            self._store_failed(experiment_id, str(e), iteration)
            return {'status': 'failed', 'error': str(e)}
    
    def _store_in_progress(self, experiment_id: str, iteration: int, timestamp: int):
        """Store in-progress experiment marker"""
        try:
            from datetime import datetime
            item = {
                'experiment_id': experiment_id,
                'timestamp': timestamp,
                'iteration': iteration,
                'status': 'in_progress',
                'started_at': datetime.now().isoformat(),
                'phase': 'Worker Processing'
            }
            
            self.table.put_item(Item=item)
            logger.info(f"✅ Marked as in_progress in DynamoDB: {experiment_id}")
            
        except Exception as e:
            logger.error(f"❌ DynamoDB in-progress marker error: {e}", exc_info=True)
    
    def _store_result(
        self,
        experiment_id: str,
        result: Dict,
        iteration: int,
        llm_reasoning: str
    ):
        """Store experiment result in DynamoDB"""
        try:
            experiment_result = result.get('result', {})
            metrics = experiment_result.get('metrics', {})
            
            # Convert floats to Decimal for DynamoDB
            item = {
                'experiment_id': experiment_id,
                'timestamp': result.get('timestamp', int(time.time())),
                'iteration': iteration,
                'status': experiment_result.get('status', 'unknown'),
                'metrics': {
                    'psnr_db': Decimal(str(metrics.get('psnr_db', 0))),
                    'ssim': Decimal(str(metrics.get('ssim', 0))),
                    'bitrate_mbps': Decimal(str(metrics.get('bitrate_mbps', 0))),
                    'compression_ratio': Decimal(str(metrics.get('compression_ratio', 0))),
                    'original_size_bytes': metrics.get('original_size_bytes', 0),
                    'compressed_size_bytes': metrics.get('compressed_size_bytes', 0)
                },
                'artifacts': {
                    'video_url': experiment_result.get('video_url'),
                    'decoder_s3_key': experiment_result.get('decoder_s3_key')
                },
                'llm_reasoning': llm_reasoning,
                'worker_id': result.get('worker_id', 'unknown')
            }
            
            self.table.put_item(Item=item)
            logger.info(f"✅ Stored result in DynamoDB: {experiment_id}")
            
        except Exception as e:
            logger.error(f"❌ DynamoDB storage error: {e}", exc_info=True)
    
    def _store_failed(self, experiment_id: str, error: str, iteration: int):
        """Store failed experiment"""
        try:
            item = {
                'experiment_id': experiment_id,
                'timestamp': int(time.time()),
                'iteration': iteration,
                'status': 'failed',
                'error': error,
                'metrics': {
                    'psnr_db': Decimal('0'),
                    'ssim': Decimal('0'),
                    'bitrate_mbps': Decimal('0'),
                    'compression_ratio': Decimal('0'),
                    'original_size_bytes': 0,
                    'compressed_size_bytes': 0
                }
            }
            
            self.table.put_item(Item=item)
            logger.info(f"✅ Stored failed result in DynamoDB: {experiment_id}")
            
        except Exception as e:
            logger.error(f"❌ DynamoDB storage error: {e}", exc_info=True)
    
    def get_recent_results(self, limit: int = 5) -> List[Dict]:
        """Get recent experiment results for LLM context"""
        try:
            response = self.table.scan(
                Limit=limit * 2  # Get more, then sort
            )
            
            items = response.get('Items', [])
            
            # Sort by timestamp, newest first
            sorted_items = sorted(
                items,
                key=lambda x: x.get('timestamp', 0),
                reverse=True
            )
            
            # Convert Decimal to float for JSON serialization
            results = []
            for item in sorted_items[:limit]:
                result = {
                    'experiment_id': item.get('experiment_id'),
                    'status': item.get('status'),
                    'iteration': item.get('iteration'),
                    'metrics': {
                        'psnr_db': float(item.get('metrics', {}).get('psnr_db', 0)),
                        'ssim': float(item.get('metrics', {}).get('ssim', 0)),
                        'compression_ratio': float(item.get('metrics', {}).get('compression_ratio', 0))
                    },
                    'error': item.get('error')
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error fetching recent results: {e}")
            return []

