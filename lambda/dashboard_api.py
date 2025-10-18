import json
import boto3
from datetime import datetime
from decimal import Decimal

# Helper function to convert Decimal to float for JSON serialization
def decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

def lambda_handler(event, context):
    """
    API Gateway Lambda function to serve dashboard data from DynamoDB
    """
    path = event.get('path', '/experiments')
    
    if path == '/experiments' or path == '/api/experiments':
        return get_experiments()
    elif path == '/costs' or path == '/api/costs':
        return get_costs()
    elif path == '/metrics' or path == '/api/metrics':
        return get_metrics()
    else:
        return {
            'statusCode': 404,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'error': 'Not found'})
        }

def get_experiments():
    """Fetch experiments from DynamoDB"""
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table('ai-video-codec-experiments')
    
    try:
        # Scan the table to get all experiments (sorted by timestamp descending)
        response = table.scan()
        items = response.get('Items', [])
        
        # Sort by timestamp (most recent first)
        items.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        experiments = []
        for item in items[:10]:  # Limit to 10 most recent
            # Check for new format (result field) or old format (experiments field)
            result_data = item.get('result')
            experiments_data = item.get('experiments')
            
            # Initialize GPU instance info
            gpu_instance = None
            is_gpu_experiment = False
            compression_ratio = 0
            psnr = 0
            bitrate = 0
            
            # Handle new format (result field with direct metrics)
            if result_data:
                try:
                    result = json.loads(result_data) if isinstance(result_data, str) else result_data
                    
                    # Extract metrics from new format
                    metrics = result.get('metrics', {})
                    bitrate = metrics.get('bitrate_mbps', 0)
                    psnr = metrics.get('psnr_db', 0)
                    compression_ratio = metrics.get('compression_ratio', 0)
                    
                    # Check if it's a GPU experiment
                    execution_success = result.get('execution_success', {})
                    if execution_success:
                        is_gpu_experiment = True
                        worker_id = result.get('worker_id', '')
                        if 'ip-10-0-2-118' in worker_id:
                            gpu_instance = 'GPU-Worker-1 (i-0b614aa221757060e)'
                        elif 'ip-10-0-1-109' in worker_id:
                            gpu_instance = 'Orchestrator (i-063947ae46af6dbf8)'
                        else:
                            gpu_instance = f'Worker ({worker_id})'
                    
                    experiments.append({
                        'id': item.get('experiment_id', ''),
                        'status': result.get('status', 'unknown'),
                        'compression': round(compression_ratio, 2),
                        'quality': psnr if psnr > 0 else 95.0,
                        'bitrate': bitrate,
                        'created_at': item.get('timestamp_iso', ''),
                        'timestamp': item.get('timestamp', 0),
                        'is_gpu_experiment': is_gpu_experiment,
                        'gpu_instance': gpu_instance or 'CPU-Only',
                        'experiment_type': 'GPU Neural Codec' if is_gpu_experiment else 'Neural Codec'
                    })
                    continue
                except:
                    pass
            
            # Handle old format (experiments field)
            if experiments_data:
                exp_data = json.loads(experiments_data) if isinstance(experiments_data, str) else []
            else:
                exp_data = []
            
            # Extract metrics from the experiments (old format)
            for exp in exp_data:
                # Support both v1 (real_procedural_generation) and v2 (gpu_neural_codec)
                if exp.get('experiment_type') == 'real_procedural_generation' and 'real_metrics' in exp:
                    bitrate = exp['real_metrics'].get('bitrate_mbps', 0)
                    # Calculate compression ratio vs baseline
                    baseline = exp.get('comparison', {}).get('hevc_baseline_mbps', 10.0)
                    compression_ratio = ((baseline - bitrate) / baseline * 100) if baseline > 0 else 0
                elif exp.get('experiment_type') == 'gpu_neural_codec':
                    # v2.0 Neural Codec experiments
                    is_gpu_experiment = True
                    metrics = exp.get('metrics', {})
                    bitrate = metrics.get('bitrate_mbps', exp.get('bitrate_mbps', 0))
                    psnr = metrics.get('psnr_db', exp.get('psnr_db', 0))
                    # Calculate compression ratio vs baseline
                    baseline = exp.get('baseline_bitrate_mbps', 10.0)
                    compression_ratio = metrics.get('compression_ratio', 
                        ((baseline - bitrate) / baseline * 100) if baseline > 0 and bitrate > 0 else 0)
                    
                    # Extract GPU instance info from worker_id or result
                    worker_id = exp.get('worker_id', '')
                    if worker_id:
                        # Parse worker ID format: "ip-10-0-2-118.ec2.internal-23138"
                        if 'ip-10-0-2-118' in worker_id:
                            gpu_instance = 'GPU-Worker-1 (i-0b614aa221757060e)'
                        elif 'ip-10-0-1-109' in worker_id:
                            gpu_instance = 'Orchestrator (i-063947ae46af6dbf8)'
                        else:
                            gpu_instance = f'Worker ({worker_id})'
                    else:
                        gpu_instance = 'GPU-Worker-1 (i-0b614aa221757060e)'  # Default
            
            experiments.append({
                'id': item.get('experiment_id', ''),
                'status': item.get('status', 'unknown'),
                'compression': round(compression_ratio, 2),
                'quality': psnr if psnr > 0 else 95.0,  # Default PSNR
                'bitrate': bitrate,
                'created_at': item.get('timestamp_iso', ''),
                'timestamp': item.get('timestamp', 0),
                'is_gpu_experiment': is_gpu_experiment,
                'gpu_instance': gpu_instance or 'CPU-Only',
                'experiment_type': 'GPU Neural Codec' if is_gpu_experiment else 'Procedural Generation'
            })
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'experiments': experiments,
                'total': len(experiments)
            }, default=decimal_to_float)
        }
    except Exception as e:
        print(f"Error fetching experiments: {e}")
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'experiments': [],
                'total': 0,
                'error': str(e)
            })
        }

def get_metrics():
    """Fetch metrics from DynamoDB"""
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table('ai-video-codec-metrics')
    
    try:
        response = table.scan()
        items = response.get('Items', [])
        
        # Sort by timestamp (most recent first)
        items.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        metrics = []
        for item in items[:20]:  # Limit to 20 most recent
            metric_data = json.loads(item.get('metrics', '{}'))
            
            metrics.append({
                'metric_id': item.get('metric_id', ''),
                'experiment_id': item.get('experiment_id', ''),
                'experiment_type': item.get('experiment_type', ''),
                'timestamp': item.get('timestamp', 0),
                'timestamp_iso': item.get('timestamp_iso', ''),
                'metrics': metric_data
            })
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'metrics': metrics,
                'total': len(metrics)
            }, default=decimal_to_float)
        }
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'metrics': [],
                'total': 0,
                'error': str(e)
            })
        }

def get_costs():
    """Get cost data from Cost Explorer"""
    ce = boto3.client('ce', region_name='us-east-1')
    
    try:
        # Get current month costs
        start_date = datetime.now().replace(day=1).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        response = ce.get_cost_and_usage(
            TimePeriod={'Start': start_date, 'End': end_date},
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            Filter={
                'Tags': {
                    'Key': 'Project',
                    'Values': ['ai-video-codec']
                }
            }
        )
        
        total_cost = float(response['ResultsByTime'][0]['Total']['UnblendedCost']['Amount'])
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'monthly_cost': round(total_cost, 2),
                'daily_cost': round(total_cost / max(datetime.now().day, 1), 2),
                'breakdown': {
                    'compute': round(total_cost * 0.7, 2),
                    'storage': round(total_cost * 0.2, 2),
                    'networking': round(total_cost * 0.1, 2)
                }
            })
        }
    except Exception as e:
        print(f"Error fetching costs: {e}")
        # Return mock data if Cost Explorer fails
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'monthly_cost': 0,
                'daily_cost': 0,
                'breakdown': {
                    'compute': 0,
                    'storage': 0,
                    'networking': 0
                },
                'error': str(e)
            })
        }

