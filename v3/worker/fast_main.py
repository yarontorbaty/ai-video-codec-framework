"""
V3.0 Fast Worker - Optimized for 10,000+ experiments/hour

HTTP server that receives batches of experiments and runs them quickly.
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import boto3
from fast_experiment_runner import FastExperimentRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS configuration
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
experiments_table = dynamodb.Table('ai-codec-v3-fast-experiments')

# Initialize runner
runner = FastExperimentRunner()


class FastWorkerHandler(BaseHTTPRequestHandler):
    """HTTP handler for fast experiment batches"""
    
    def do_POST(self):
        """Handle batch experiment request"""
        if self.path == '/batch':
            self._handle_batch()
        else:
            self.send_error(404)
    
    def _handle_batch(self):
        """Process a batch of experiments"""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            request = json.loads(body)
            
            experiments = request.get('experiments', [])
            generation = request.get('generation', 0)  # Support evolutionary mode
            logger.info(f"🚀 Received batch: {len(experiments)} experiments (generation {generation})")
            
            # Run batch
            results = runner.run_batch(experiments)
            
            # Add generation to all results
            for result in results:
                result['generation'] = generation
            
            # Store results in DynamoDB (batch write)
            self._store_results_batch(results)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'success': True,
                'total': len(results),
                'succeeded': sum(1 for r in results if r['status'] == 'success'),
                'failed': sum(1 for r in results if r['status'] == 'failed')
            }).encode())
            
        except Exception as e:
            logger.error(f"❌ Batch error: {e}", exc_info=True)
            self.send_error(500, str(e))
    
    def _store_results_batch(self, results: list):
        """Store experiment results in DynamoDB (batch operation)"""
        import time
        import math
        from decimal import Decimal
        
        # DynamoDB batch write (25 items at a time)
        batch_size = 25
        for i in range(0, len(results), batch_size):
            batch = results[i:i+batch_size]
            
            with experiments_table.batch_writer() as writer:
                for result in batch:
                    # Build metrics object for dashboard compatibility
                    metrics = {}
                    
                    if result.get('mse') is not None and result['status'] == 'success':
                        mse = float(result['mse'])
                        compression_ratio = float(result.get('compression_ratio', 1.0))
                        compressed_size = int(result.get('compressed_size', 0))
                        
                        # Calculate PSNR from MSE
                        if mse > 0:
                            psnr_db = 10 * math.log10((255.0 ** 2) / mse)
                        else:
                            psnr_db = 100.0
                        
                        # Estimate SSIM from PSNR (rough heuristic)
                        if psnr_db >= 40: ssim = 0.99
                        elif psnr_db >= 35: ssim = 0.95
                        elif psnr_db >= 30: ssim = 0.85
                        elif psnr_db >= 25: ssim = 0.75
                        else: ssim = 0.5
                        
                        # Calculate bitrate (assuming 10 frames @ 30fps = 0.33s)
                        video_duration_seconds = 10 / 30.0
                        bitrate_mbps = (compressed_size * 8) / (video_duration_seconds * 1000 * 1000)
                        
                        metrics = {
                            'psnr_db': Decimal(str(round(psnr_db, 2))),
                            'ssim': Decimal(str(round(ssim, 4))),
                            'mse': Decimal(str(round(mse, 2))),
                            'compression_ratio': Decimal(str(round(compression_ratio, 2))),
                            'compressed_size_bytes': Decimal(str(compressed_size)),
                            'bitrate_mbps': Decimal(str(round(bitrate_mbps, 3))),
                            'encoding_time_ms': Decimal(str(result.get('time_ms', 0)))
                        }
                    
                    # Convert floats to Decimal for DynamoDB
                    item = {
                        'experiment_id': result['experiment_id'],
                        'timestamp': int(time.time()),
                        'generation': result.get('generation', 0),  # Track evolutionary generation
                        'status': result['status']
                    }
                    
                    # Add metrics if available
                    if metrics:
                        item['metrics'] = metrics
                    
                    # Add error for failed experiments
                    if result.get('error'):
                        item['error'] = result['error']
                    
                    writer.put_item(Item=item)
        
        logger.info(f"💾 Stored {len(results)} results in DynamoDB")
    
    def do_GET(self):
        """Health check"""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'healthy'}).encode())
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"{self.address_string()} - {format % args}")


def main():
    """Start fast worker HTTP server"""
    PORT = 8080
    
    logger.info("=" * 60)
    logger.info("🚀 Fast Worker v3.0 Starting")
    logger.info("=" * 60)
    logger.info(f"Port: {PORT}")
    logger.info(f"DynamoDB Table: {experiments_table.table_name}")
    logger.info(f"Test Video: {runner.test_frames[0].shape} x {len(runner.test_frames)} frames")
    logger.info("=" * 60)
    
    server = HTTPServer(('0.0.0.0', PORT), FastWorkerHandler)
    
    try:
        logger.info("✅ Fast worker ready - waiting for batches...")
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\n⏹️  Shutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()

