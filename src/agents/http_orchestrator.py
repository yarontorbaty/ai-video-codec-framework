#!/usr/bin/env python3
"""
HTTP-based Neural Codec Orchestrator
Manages neural codec experiments using direct HTTP communication with workers.
No SQS complexity - simple, reliable HTTP requests.
"""

import os
import sys
import json
import time
import logging
import traceback
import boto3
import requests
from datetime import datetime
from typing import Dict, List, Optional
from flask import Flask, request, jsonify
import threading

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS clients
s3 = boto3.client('s3', region_name='us-east-1')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

# Configuration
EXPERIMENTS_TABLE = 'ai-video-codec-experiments'
HTTP_PORT = int(os.environ.get('ORCHESTRATOR_HTTP_PORT', 8081))
WORKER_URLS = os.environ.get('WORKER_URLS', 'http://localhost:8080').split(',')

# Flask app
app = Flask(__name__)

class HTTPOrchestrator:
    """HTTP-based Neural Codec Orchestrator"""
    
    def __init__(self):
        self.worker_urls = [url.strip() for url in WORKER_URLS]
        self.active_experiments = {}
        self.experiment_results = {}
        
    def get_available_worker(self) -> Optional[str]:
        """Find an available worker."""
        for worker_url in self.worker_urls:
            try:
                response = requests.get(f"{worker_url}/status", timeout=5)
                if response.status_code == 200:
                    status = response.json()
                    if not status.get('is_processing', False):
                        return worker_url
            except Exception as e:
                logger.warning(f"Worker {worker_url} not available: {e}")
        
        return None
    
    def send_experiment_to_worker(self, worker_url: str, job_data: Dict) -> bool:
        """Send experiment to a specific worker."""
        try:
            logger.info(f"📤 Sending experiment to worker: {worker_url}")
            
            response = requests.post(
                f"{worker_url}/experiment",
                json=job_data,
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get('status') == 'accepted':
                logger.info(f"✅ Experiment accepted by worker: {worker_url}")
                return True
            else:
                logger.warning(f"⚠️  Worker declined experiment: {result}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to send experiment to {worker_url}: {e}")
            return False
    
    def create_experiment(self, experiment_data: Dict) -> str:
        """Create a new experiment."""
        try:
            experiment_id = f"exp_{int(time.time())}"
            
            # Store in DynamoDB
            table = dynamodb.Table(EXPERIMENTS_TABLE)
            table.put_item(Item={
                'experiment_id': experiment_id,
                'timestamp': int(time.time()),
                'timestamp_iso': datetime.utcnow().isoformat(),
                'status': 'created',
                'experiments': json.dumps([experiment_data]),
                'created_by': 'http_orchestrator'
            })
            
            logger.info(f"📝 Created experiment: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create experiment: {e}")
            raise
    
    def dispatch_experiment(self, experiment_data: Dict) -> Dict:
        """Dispatch experiment to available worker."""
        try:
            experiment_id = self.create_experiment(experiment_data)
            
            # Prepare job data
            job_data = {
                'experiment_id': experiment_id,
                'encoding_agent_code': experiment_data.get('encoding_agent_code', ''),
                'decoding_agent_code': experiment_data.get('decoding_agent_code', ''),
                'hypothesis': experiment_data.get('hypothesis', ''),
                'compression_strategy': experiment_data.get('compression_strategy', ''),
                'expected_bitrate_mbps': experiment_data.get('expected_bitrate_mbps', 1.0),
                'timestamp': int(time.time())
            }
            
            # Find available worker
            worker_url = self.get_available_worker()
            if not worker_url:
                return {
                    'status': 'failed',
                    'error': 'No available workers',
                    'experiment_id': experiment_id
                }
            
            # Send to worker
            success = self.send_experiment_to_worker(worker_url, job_data)
            if success:
                self.active_experiments[experiment_id] = {
                    'worker_url': worker_url,
                    'start_time': time.time(),
                    'data': experiment_data
                }
                
                return {
                    'status': 'dispatched',
                    'experiment_id': experiment_id,
                    'worker_url': worker_url,
                    'message': 'Experiment sent to worker'
                }
            else:
                return {
                    'status': 'failed',
                    'error': 'Failed to send to worker',
                    'experiment_id': experiment_id
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to dispatch experiment: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def handle_experiment_result(self, result: Dict):
        """Handle result from worker."""
        try:
            experiment_id = result.get('experiment_id')
            if not experiment_id:
                logger.error("❌ No experiment_id in result")
                return
            
            logger.info(f"📥 Received result for experiment: {experiment_id}")
            
            # Remove from active experiments
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
            
            # Store result
            self.experiment_results[experiment_id] = result
            
            # Update DynamoDB - need both experiment_id and timestamp as key
            table = dynamodb.Table(EXPERIMENTS_TABLE)
            
            # First, get the original timestamp from active experiments or result
            original_timestamp = None
            if experiment_id in self.active_experiments:
                original_timestamp = self.active_experiments[experiment_id]['start_time']
            elif 'timestamp' in result:
                original_timestamp = result['timestamp']
            else:
                # Fallback: use current time
                original_timestamp = int(time.time())
            
            table.update_item(
                Key={
                    'experiment_id': experiment_id,
                    'timestamp': int(original_timestamp)
                },
                UpdateExpression='SET #status = :status, #result = :result, #updated_at = :updated_at',
                ExpressionAttributeNames={
                    '#status': 'status',
                    '#result': 'result',
                    '#updated_at': 'updated_at'
                },
                ExpressionAttributeValues={
                    ':status': result.get('status', 'completed'),
                    ':result': json.dumps(result),
                    ':updated_at': int(time.time())
                }
            )
            
            logger.info(f"✅ Updated experiment {experiment_id} with result")
            
        except Exception as e:
            logger.error(f"❌ Failed to handle experiment result: {e}")
    
    def get_experiment_status(self, experiment_id: str) -> Dict:
        """Get status of an experiment."""
        if experiment_id in self.active_experiments:
            return {
                'status': 'processing',
                'worker_url': self.active_experiments[experiment_id]['worker_url'],
                'start_time': self.active_experiments[experiment_id]['start_time']
            }
        elif experiment_id in self.experiment_results:
            return self.experiment_results[experiment_id]
        else:
            return {'status': 'not_found'}

# Global orchestrator instance
orchestrator = HTTPOrchestrator()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    available_workers = []
    for worker_url in orchestrator.worker_urls:
        try:
            response = requests.get(f"{worker_url}/health", timeout=5)
            if response.status_code == 200:
                available_workers.append({
                    'url': worker_url,
                    'status': response.json()
                })
        except:
            available_workers.append({
                'url': worker_url,
                'status': 'unavailable'
            })
    
    return jsonify({
        'status': 'healthy',
        'active_experiments': len(orchestrator.active_experiments),
        'completed_experiments': len(orchestrator.experiment_results),
        'available_workers': available_workers,
        'timestamp': int(time.time())
    })

@app.route('/experiment', methods=['POST'])
def create_experiment():
    """Create and dispatch a new experiment."""
    try:
        experiment_data = request.get_json()
        if not experiment_data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        result = orchestrator.dispatch_experiment(experiment_data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/experiment_result', methods=['POST'])
def handle_result():
    """Handle experiment result from worker."""
    try:
        result = request.get_json()
        if not result:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        orchestrator.handle_experiment_result(result)
        return jsonify({'status': 'received'})
        
    except Exception as e:
        logger.error(f"Error handling result: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/experiment/<experiment_id>/status', methods=['GET'])
def get_experiment_status(experiment_id: str):
    """Get status of an experiment."""
    try:
        status = orchestrator.get_experiment_status(experiment_id)
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting experiment status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/experiments', methods=['GET'])
def list_experiments():
    """List recent experiments."""
    try:
        return jsonify({
            'active_experiments': list(orchestrator.active_experiments.keys()),
            'completed_experiments': list(orchestrator.experiment_results.keys()),
            'timestamp': int(time.time())
        })
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        return jsonify({'error': str(e)}), 500

def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("🚀 HTTP NEURAL CODEC ORCHESTRATOR STARTING")
    logger.info("=" * 80)
    logger.info(f"   HTTP Port: {HTTP_PORT}")
    logger.info(f"   Worker URLs: {orchestrator.worker_urls}")
    logger.info("=" * 80)
    
    logger.info(f"🌐 Starting HTTP server on port {HTTP_PORT}")
    logger.info("📡 Ready to accept experiment requests!")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=HTTP_PORT, debug=False)

if __name__ == '__main__':
    main()
