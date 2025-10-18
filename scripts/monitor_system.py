#!/usr/bin/env python3
"""
Neural Codec HTTP System Monitor
Continuously monitors the health and performance of the HTTP-based system.
"""

import requests
import json
import time
import boto3
from datetime import datetime, timedelta
import sys

# Configuration
WORKER_URL = "http://18.208.180.67:8080"
ORCHESTRATOR_URL = "http://34.239.1.29:8081"
MONITORING_INTERVAL = 60  # seconds

class SystemMonitor:
    """Monitor the neural codec HTTP system."""
    
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
        self.metrics_buffer = []
        
    def check_worker_health(self):
        """Check worker health and return status."""
        try:
            start_time = time.time()
            response = requests.get(f"{WORKER_URL}/health", timeout=10)
            response_time = (time.time() - start_time) * 1000  # ms
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'status': 'healthy',
                    'response_time_ms': response_time,
                    'worker_id': data.get('worker_id'),
                    'device': data.get('device'),
                    'jobs_processed': data.get('jobs_processed', 0)
                }
            else:
                return {
                    'status': 'unhealthy',
                    'response_time_ms': response_time,
                    'error': f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                'status': 'unreachable',
                'response_time_ms': 0,
                'error': str(e)
            }
    
    def check_orchestrator_health(self):
        """Check orchestrator health and return status."""
        try:
            start_time = time.time()
            response = requests.get(f"{ORCHESTRATOR_URL}/health", timeout=10)
            response_time = (time.time() - start_time) * 1000  # ms
            
            if response.status_code == 200:
                data = response.json()
                available_workers = len([w for w in data.get('available_workers', []) if w.get('status', {}).get('status') == 'healthy'])
                
                return {
                    'status': 'healthy',
                    'response_time_ms': response_time,
                    'active_experiments': data.get('active_experiments', 0),
                    'completed_experiments': data.get('completed_experiments', 0),
                    'available_workers': available_workers
                }
            else:
                return {
                    'status': 'unhealthy',
                    'response_time_ms': response_time,
                    'error': f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                'status': 'unreachable',
                'response_time_ms': 0,
                'error': str(e)
            }
    
    def send_custom_metrics(self, worker_status, orchestrator_status):
        """Send custom metrics to CloudWatch."""
        try:
            timestamp = datetime.utcnow()
            
            metrics = []
            
            # Worker metrics
            if worker_status['status'] == 'healthy':
                metrics.extend([
                    {
                        'MetricName': 'WorkerHealth',
                        'Value': 1,
                        'Unit': 'Count',
                        'Timestamp': timestamp,
                        'Dimensions': [{'Name': 'Service', 'Value': 'Worker'}]
                    },
                    {
                        'MetricName': 'WorkerResponseTime',
                        'Value': worker_status['response_time_ms'],
                        'Unit': 'Milliseconds',
                        'Timestamp': timestamp,
                        'Dimensions': [{'Name': 'Service', 'Value': 'Worker'}]
                    },
                    {
                        'MetricName': 'JobsProcessed',
                        'Value': worker_status.get('jobs_processed', 0),
                        'Unit': 'Count',
                        'Timestamp': timestamp,
                        'Dimensions': [{'Name': 'Service', 'Value': 'Worker'}]
                    }
                ])
            else:
                metrics.append({
                    'MetricName': 'WorkerHealth',
                    'Value': 0,
                    'Unit': 'Count',
                    'Timestamp': timestamp,
                    'Dimensions': [{'Name': 'Service', 'Value': 'Worker'}]
                })
            
            # Orchestrator metrics
            if orchestrator_status['status'] == 'healthy':
                metrics.extend([
                    {
                        'MetricName': 'OrchestratorHealth',
                        'Value': 1,
                        'Unit': 'Count',
                        'Timestamp': timestamp,
                        'Dimensions': [{'Name': 'Service', 'Value': 'Orchestrator'}]
                    },
                    {
                        'MetricName': 'OrchestratorResponseTime',
                        'Value': orchestrator_status['response_time_ms'],
                        'Unit': 'Milliseconds',
                        'Timestamp': timestamp,
                        'Dimensions': [{'Name': 'Service', 'Value': 'Orchestrator'}]
                    },
                    {
                        'MetricName': 'ActiveExperiments',
                        'Value': orchestrator_status.get('active_experiments', 0),
                        'Unit': 'Count',
                        'Timestamp': timestamp,
                        'Dimensions': [{'Name': 'Service', 'Value': 'Orchestrator'}]
                    },
                    {
                        'MetricName': 'AvailableWorkers',
                        'Value': orchestrator_status.get('available_workers', 0),
                        'Unit': 'Count',
                        'Timestamp': timestamp,
                        'Dimensions': [{'Name': 'Service', 'Value': 'Orchestrator'}]
                    }
                ])
            else:
                metrics.append({
                    'MetricName': 'OrchestratorHealth',
                    'Value': 0,
                    'Unit': 'Count',
                    'Timestamp': timestamp,
                    'Dimensions': [{'Name': 'Service', 'Value': 'Orchestrator'}]
                })
            
            # Send metrics in batches
            for i in range(0, len(metrics), 20):
                batch = metrics[i:i+20]
                self.cloudwatch.put_metric_data(
                    Namespace='NeuralCodec/HTTP',
                    MetricData=batch
                )
            
            print(f"📊 Sent {len(metrics)} metrics to CloudWatch")
            
        except Exception as e:
            print(f"❌ Failed to send metrics to CloudWatch: {e}")
    
    def log_status(self, worker_status, orchestrator_status):
        """Log current system status."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n🔍 [{timestamp}] System Status Check:")
        print(f"   Worker: {worker_status['status']} ({worker_status.get('response_time_ms', 0):.1f}ms)")
        if worker_status.get('error'):
            print(f"      Error: {worker_status['error']}")
        if worker_status.get('jobs_processed') is not None:
            print(f"      Jobs Processed: {worker_status['jobs_processed']}")
        
        print(f"   Orchestrator: {orchestrator_status['status']} ({orchestrator_status.get('response_time_ms', 0):.1f}ms)")
        if orchestrator_status.get('error'):
            print(f"      Error: {orchestrator_status['error']}")
        if orchestrator_status.get('available_workers') is not None:
            print(f"      Available Workers: {orchestrator_status['available_workers']}")
        if orchestrator_status.get('active_experiments') is not None:
            print(f"      Active Experiments: {orchestrator_status['active_experiments']}")
    
    def run_monitoring_cycle(self):
        """Run a single monitoring cycle."""
        print(f"🔍 Starting monitoring cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check service health
        worker_status = self.check_worker_health()
        orchestrator_status = self.check_orchestrator_health()
        
        # Log status
        self.log_status(worker_status, orchestrator_status)
        
        # Send metrics to CloudWatch
        self.send_custom_metrics(worker_status, orchestrator_status)
        
        # Check for critical issues
        critical_issues = []
        if worker_status['status'] != 'healthy':
            critical_issues.append(f"Worker unhealthy: {worker_status.get('error', 'Unknown error')}")
        if orchestrator_status['status'] != 'healthy':
            critical_issues.append(f"Orchestrator unhealthy: {orchestrator_status.get('error', 'Unknown error')}")
        if orchestrator_status.get('available_workers', 0) == 0:
            critical_issues.append("No available workers")
        
        if critical_issues:
            print(f"🚨 CRITICAL ISSUES DETECTED:")
            for issue in critical_issues:
                print(f"   - {issue}")
            return False
        else:
            print(f"✅ All systems healthy")
            return True
    
    def run_continuous_monitoring(self, duration_minutes=None):
        """Run continuous monitoring."""
        print(f"🚀 Starting continuous monitoring...")
        print(f"   Monitoring interval: {MONITORING_INTERVAL} seconds")
        if duration_minutes:
            print(f"   Duration: {duration_minutes} minutes")
        print(f"   Worker URL: {WORKER_URL}")
        print(f"   Orchestrator URL: {ORCHESTRATOR_URL}")
        
        start_time = time.time()
        cycle_count = 0
        healthy_cycles = 0
        
        try:
            while True:
                cycle_count += 1
                print(f"\n{'='*60}")
                print(f"📊 Monitoring Cycle #{cycle_count}")
                
                # Run monitoring cycle
                is_healthy = self.run_monitoring_cycle()
                if is_healthy:
                    healthy_cycles += 1
                
                # Check if we should stop
                if duration_minutes:
                    elapsed_minutes = (time.time() - start_time) / 60
                    if elapsed_minutes >= duration_minutes:
                        break
                
                # Wait for next cycle
                print(f"⏳ Waiting {MONITORING_INTERVAL} seconds until next check...")
                time.sleep(MONITORING_INTERVAL)
                
        except KeyboardInterrupt:
            print(f"\n⏹️  Monitoring stopped by user")
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"📊 Monitoring Summary:")
        print(f"   Total cycles: {cycle_count}")
        print(f"   Healthy cycles: {healthy_cycles}")
        print(f"   Health rate: {(healthy_cycles/cycle_count)*100:.1f}%")
        print(f"   Duration: {(time.time() - start_time)/60:.1f} minutes")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Neural Codec HTTP System')
    parser.add_argument('--duration', type=int, help='Monitoring duration in minutes (default: continuous)')
    parser.add_argument('--once', action='store_true', help='Run single monitoring cycle and exit')
    
    args = parser.parse_args()
    
    monitor = SystemMonitor()
    
    if args.once:
        # Single monitoring cycle
        success = monitor.run_monitoring_cycle()
        sys.exit(0 if success else 1)
    else:
        # Continuous monitoring
        monitor.run_continuous_monitoring(duration_minutes=args.duration)

if __name__ == '__main__':
    main()
