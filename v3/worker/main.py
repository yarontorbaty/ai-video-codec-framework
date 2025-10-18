"""
V3.0 GPU Worker - Main HTTP Server

Simple HTTP server that receives experiment requests, executes video compression
experiments, calculates real metrics, and stores results.
"""

import json
import os
import time
import logging
import shutil
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any

from experiment_runner import ExperimentRunner
from metrics_calculator import MetricsCalculator
from s3_uploader import S3Uploader

# Configuration
PORT = 8080
WORKER_ID = os.getenv('WORKER_ID', 'worker-1')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for experiment requests"""
    
    runner = ExperimentRunner()
    metrics = MetricsCalculator()
    uploader = S3Uploader()
    
    def do_POST(self):
        """Handle POST requests for experiments"""
        if self.path == '/experiment':
            self._handle_experiment()
        elif self.path == '/health':
            self._handle_health()
        else:
            self.send_error(404, f"Unknown path: {self.path}")
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            self._handle_health()
        elif self.path == '/status':
            self._handle_status()
        else:
            self.send_error(404, f"Unknown path: {self.path}")
    
    def _handle_experiment(self):
        """Execute an experiment"""
        try:
            # Parse request
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            experiment = json.loads(body.decode('utf-8'))
            
            experiment_id = experiment.get('experiment_id')
            encoding_code = experiment.get('encoding_code', '')
            decoding_code = experiment.get('decoding_code', '')
            
            logger.info(f"🎯 Received experiment: {experiment_id}")
            logger.info(f"   Encoding code: {len(encoding_code)} bytes")
            logger.info(f"   Decoding code: {len(decoding_code)} bytes")
            
            if not encoding_code or not decoding_code:
                self._send_json_response({
                    'status': 'error',
                    'error': 'Missing encoding or decoding code'
                }, 400)
                return
            
            # Run experiment
            start_time = time.time()
            result = self.runner.run_experiment(
                experiment_id=experiment_id,
                encoding_code=encoding_code,
                decoding_code=decoding_code
            )
            runtime = time.time() - start_time
            
            # Calculate metrics if successful
            if result['status'] == 'success':
                logger.info(f"✅ Experiment succeeded, calculating metrics...")
                metrics = self.metrics.calculate_metrics(
                    original_path=result['original_path'],
                    compressed_path=result['compressed_path'],
                    reconstructed_path=result['reconstructed_path']
                )
                result['metrics'] = metrics
                result['runtime_seconds'] = runtime
                
                # Upload artifacts
                logger.info(f"📤 Uploading artifacts to S3...")
                video_url = self.uploader.upload_video(
                    result['reconstructed_path'],
                    experiment_id
                )
                decoder_key = self.uploader.save_decoder(
                    decoding_code,
                    experiment_id
                )
                
                result['video_url'] = video_url
                result['decoder_s3_key'] = decoder_key
                
                logger.info(f"✅ Experiment complete: PSNR={metrics['psnr_db']:.2f}dB, SSIM={metrics['ssim']:.3f}")
            else:
                logger.error(f"❌ Experiment failed: {result.get('error')}")
            
            # Clean up temporary files
            self._cleanup_temp_files(result)
            
            # Send response
            self._send_json_response({
                'status': 'completed',
                'experiment_id': experiment_id,
                'result': result,
                'worker_id': WORKER_ID,
                'timestamp': int(time.time())
            })
            
        except Exception as e:
            logger.error(f"❌ Error processing experiment: {e}", exc_info=True)
            self._send_json_response({
                'status': 'error',
                'error': str(e)
            }, 500)
    
    def _cleanup_temp_files(self, result: Dict[str, Any]):
        """Clean up temporary experiment files to prevent disk full"""
        try:
            # Get temp directory from any of the paths
            temp_dir = None
            for key in ['original_path', 'compressed_path', 'reconstructed_path']:
                if result.get(key):
                    temp_dir = os.path.dirname(result[key])
                    break
            
            if temp_dir and os.path.exists(temp_dir):
                logger.info(f"🧹 Cleaning up temp directory: {temp_dir}")
                shutil.rmtree(temp_dir)
                logger.info(f"✅ Temp directory cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Failed to clean up temp files: {e}")
    
    def _handle_health(self):
        """Health check endpoint"""
        self._send_json_response({
            'status': 'healthy',
            'worker_id': WORKER_ID,
            'timestamp': int(time.time())
        })
    
    def _handle_status(self):
        """Status endpoint"""
        self._send_json_response({
            'status': 'running',
            'worker_id': WORKER_ID,
            'experiments_completed': 0,  # TODO: track this
            'timestamp': int(time.time())
        })
    
    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override to use logger instead of print"""
        logger.info(f"{self.client_address[0]} - {format%args}")


def main():
    """Start the worker HTTP server"""
    logger.info("="*80)
    logger.info("🚀 AI Video Codec Worker v3.0 Starting")
    logger.info("="*80)
    logger.info(f"   Worker ID: {WORKER_ID}")
    logger.info(f"   Port: {PORT}")
    logger.info(f"   Log Level: {LOG_LEVEL}")
    logger.info("="*80)
    
    server = HTTPServer(('0.0.0.0', PORT), WorkerHandler)
    
    logger.info(f"✅ Worker listening on port {PORT}")
    logger.info("📡 Ready to accept experiment requests")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down worker...")
        server.shutdown()


if __name__ == '__main__':
    main()

