"""
Health Monitor Lambda
Monitors the AI Video Codec orchestrator and auto-heals when needed
"""
import json
import boto3
import time
from datetime import datetime, timedelta

# AWS clients
ssm = boto3.client('ssm')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
cloudwatch = boto3.client('cloudwatch')
sns = boto3.client('sns')

# Configuration
ORCHESTRATOR_INSTANCE_ID = 'i-063947ae46af6dbf8'
EXPERIMENTS_TABLE = 'ai-video-codec-experiments'
METRICS_TABLE = 'ai-video-codec-metrics'
CONTROL_TABLE = 'ai-video-codec-control'
HEALTH_CHECK_NAMESPACE = 'AiVideoCodec/Orchestrator'


def lambda_handler(event, context):
    """
    Main handler - checks orchestrator health and auto-heals if needed
    """
    print(f"[{datetime.utcnow().isoformat()}] Starting health check...")
    
    health_status = {
        'timestamp': datetime.utcnow().isoformat(),
        'orchestrator_running': False,
        'recent_experiments': False,
        'disk_space_ok': True,
        'memory_ok': True,
        'action_taken': None,
        'healthy': False
    }
    
    try:
        # Check 1: Is the orchestrator process running?
        orchestrator_running = check_orchestrator_process()
        health_status['orchestrator_running'] = orchestrator_running
        
        # Check 2: Have there been recent experiments? (within last 10 minutes)
        recent_experiments = check_recent_experiments()
        health_status['recent_experiments'] = recent_experiments
        
        # Check 3: System resources
        resources = check_system_resources()
        health_status['disk_space_ok'] = resources['disk_ok']
        health_status['memory_ok'] = resources['memory_ok']
        health_status['cpu_usage'] = resources['cpu_usage']
        health_status['memory_usage'] = resources['memory_usage']
        health_status['disk_usage'] = resources['disk_usage']
        
        # Determine overall health
        health_status['healthy'] = (
            orchestrator_running and 
            health_status['disk_space_ok'] and 
            health_status['memory_ok']
        )
        
        # Auto-heal if needed
        if not health_status['healthy']:
            print(f"❌ Orchestrator unhealthy: {health_status}")
            heal_action = auto_heal(health_status)
            health_status['action_taken'] = heal_action
        else:
            print(f"✅ Orchestrator healthy: {health_status}")
        
        # Log to DynamoDB control table
        log_health_status(health_status)
        
        # Send metrics to CloudWatch
        send_cloudwatch_metrics(health_status)
        
        return {
            'statusCode': 200,
            'body': json.dumps(health_status)
        }
        
    except Exception as e:
        print(f"❌ Health check error: {e}")
        import traceback
        traceback.print_exc()
        
        health_status['error'] = str(e)
        health_status['action_taken'] = 'error'
        
        return {
            'statusCode': 500,
            'body': json.dumps(health_status)
        }


def check_orchestrator_process():
    """Check if the orchestrator script is running"""
    try:
        response = ssm.send_command(
            InstanceIds=[ORCHESTRATOR_INSTANCE_ID],
            DocumentName='AWS-RunShellScript',
            Parameters={
                'commands': [
                    'ps aux | grep autonomous_orchestrator_llm.sh | grep -v grep | wc -l'
                ]
            }
        )
        
        command_id = response['Command']['CommandId']
        time.sleep(2)  # Wait for command to execute
        
        result = ssm.get_command_invocation(
            CommandId=command_id,
            InstanceId=ORCHESTRATOR_INSTANCE_ID
        )
        
        output = result.get('StandardOutputContent', '0').strip()
        process_count = int(output)
        
        return process_count > 0
        
    except Exception as e:
        print(f"Error checking orchestrator process: {e}")
        return False


def check_recent_experiments():
    """Check if there have been recent experiments"""
    try:
        table = dynamodb.Table(EXPERIMENTS_TABLE)
        
        # Check for experiments in the last 10 minutes
        ten_minutes_ago = int((datetime.utcnow() - timedelta(minutes=10)).timestamp())
        
        response = table.scan(
            Limit=50
        )
        
        recent_count = 0
        for item in response.get('Items', []):
            timestamp = item.get('timestamp', 0)
            if timestamp > ten_minutes_ago:
                recent_count += 1
        
        print(f"Found {recent_count} experiments in the last 10 minutes")
        return recent_count > 0
        
    except Exception as e:
        print(f"Error checking recent experiments: {e}")
        return False


def check_system_resources():
    """Check system resource usage"""
    try:
        response = ssm.send_command(
            InstanceIds=[ORCHESTRATOR_INSTANCE_ID],
            DocumentName='AWS-RunShellScript',
            Parameters={
                'commands': [
                    'echo "CPU:$(top -bn1 | grep "Cpu(s)" | awk \'{print $2}\' | cut -d"%" -f1)"',
                    'echo "MEMORY:$(free | grep Mem | awk \'{printf("%.2f", ($3/$2) * 100.0)}\')"',
                    'echo "DISK:$(df -h / | tail -1 | awk \'{print $5}\' | sed \'s/%//\')"'
                ]
            }
        )
        
        command_id = response['Command']['CommandId']
        time.sleep(2)
        
        result = ssm.get_command_invocation(
            CommandId=command_id,
            InstanceId=ORCHESTRATOR_INSTANCE_ID
        )
        
        output = result.get('StandardOutputContent', '')
        
        # Parse output
        resources = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'disk_ok': True,
            'memory_ok': True
        }
        
        for line in output.split('\n'):
            if line.startswith('CPU:'):
                resources['cpu_usage'] = float(line.split(':')[1])
            elif line.startswith('MEMORY:'):
                resources['memory_usage'] = float(line.split(':')[1])
            elif line.startswith('DISK:'):
                resources['disk_usage'] = float(line.split(':')[1])
        
        # Check thresholds
        resources['disk_ok'] = resources['disk_usage'] < 90
        resources['memory_ok'] = resources['memory_usage'] < 95
        
        return resources
        
    except Exception as e:
        print(f"Error checking system resources: {e}")
        return {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'disk_ok': True,
            'memory_ok': True
        }


def auto_heal(health_status):
    """Auto-heal the orchestrator based on health status"""
    try:
        actions = []
        
        # If disk is full, clean up old logs and temp files
        if not health_status['disk_space_ok']:
            print("🔧 Cleaning up disk space...")
            cleanup_disk()
            actions.append('disk_cleanup')
        
        # If orchestrator is not running, restart it
        if not health_status['orchestrator_running']:
            print("🔧 Restarting orchestrator...")
            restart_orchestrator()
            actions.append('orchestrator_restart')
        
        # If orchestrator is running but no recent experiments, restart it
        elif not health_status['recent_experiments']:
            print("🔧 Orchestrator stuck - restarting...")
            restart_orchestrator()
            actions.append('orchestrator_restart_stuck')
        
        return ','.join(actions) if actions else 'none'
        
    except Exception as e:
        print(f"Error during auto-heal: {e}")
        return f'error:{str(e)}'


def cleanup_disk():
    """Clean up disk space"""
    try:
        ssm.send_command(
            InstanceIds=[ORCHESTRATOR_INSTANCE_ID],
            DocumentName='AWS-RunShellScript',
            Parameters={
                'commands': [
                    'find /tmp -type f -mtime +7 -delete',
                    'journalctl --vacuum-time=7d',
                    'rm -f /home/ec2-user/ai-video-codec/*.mp4',
                    'rm -f /tmp/*.mp4'
                ]
            }
        )
        print("✅ Disk cleanup initiated")
    except Exception as e:
        print(f"Error cleaning up disk: {e}")


def restart_orchestrator():
    """Restart the orchestrator"""
    try:
        # Kill existing process
        ssm.send_command(
            InstanceIds=[ORCHESTRATOR_INSTANCE_ID],
            DocumentName='AWS-RunShellScript',
            Parameters={
                'commands': [
                    'pkill -f autonomous_orchestrator_llm.sh || true',
                    'sleep 2'
                ]
            }
        )
        
        time.sleep(3)
        
        # Start new process
        response = ssm.send_command(
            InstanceIds=[ORCHESTRATOR_INSTANCE_ID],
            DocumentName='AWS-RunShellScript',
            Parameters={
                'commands': [
                    'cd /home/ec2-user/ai-video-codec',
                    'nohup bash scripts/autonomous_orchestrator_llm.sh > /tmp/orch.log 2>&1 &',
                    'sleep 2',
                    'ps aux | grep autonomous_orchestrator_llm.sh | grep -v grep'
                ]
            }
        )
        
        print(f"✅ Orchestrator restart initiated: {response['Command']['CommandId']}")
        
    except Exception as e:
        print(f"Error restarting orchestrator: {e}")


def log_health_status(status):
    """Log health status to DynamoDB control table"""
    try:
        table = dynamodb.Table(CONTROL_TABLE)
        
        table.put_item(
            Item={
                'control_id': f"health_check_{int(time.time())}",
                'timestamp': int(time.time()),
                'status': json.dumps(status),
                'type': 'health_check'
            }
        )
        
    except Exception as e:
        print(f"Error logging health status: {e}")


def send_cloudwatch_metrics(health_status):
    """Send metrics to CloudWatch"""
    try:
        metrics = [
            {
                'MetricName': 'OrchestratorRunning',
                'Value': 1.0 if health_status['orchestrator_running'] else 0.0,
                'Unit': 'None',
                'Timestamp': datetime.utcnow()
            },
            {
                'MetricName': 'RecentExperiments',
                'Value': 1.0 if health_status['recent_experiments'] else 0.0,
                'Unit': 'None',
                'Timestamp': datetime.utcnow()
            },
            {
                'MetricName': 'OverallHealth',
                'Value': 1.0 if health_status['healthy'] else 0.0,
                'Unit': 'None',
                'Timestamp': datetime.utcnow()
            }
        ]
        
        # Add resource metrics if available
        if 'cpu_usage' in health_status:
            metrics.append({
                'MetricName': 'CPUUsage',
                'Value': health_status['cpu_usage'],
                'Unit': 'Percent',
                'Timestamp': datetime.utcnow()
            })
        
        if 'memory_usage' in health_status:
            metrics.append({
                'MetricName': 'MemoryUsage',
                'Value': health_status['memory_usage'],
                'Unit': 'Percent',
                'Timestamp': datetime.utcnow()
            })
        
        if 'disk_usage' in health_status:
            metrics.append({
                'MetricName': 'DiskUsage',
                'Value': health_status['disk_usage'],
                'Unit': 'Percent',
                'Timestamp': datetime.utcnow()
            })
        
        cloudwatch.put_metric_data(
            Namespace=HEALTH_CHECK_NAMESPACE,
            MetricData=metrics
        )
        
        print(f"✅ Sent {len(metrics)} metrics to CloudWatch")
        
    except Exception as e:
        print(f"Error sending CloudWatch metrics: {e}")

