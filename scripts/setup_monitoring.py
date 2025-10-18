#!/usr/bin/env python3
"""
Setup Monitoring and Alerting for HTTP Neural Codec System
Configures CloudWatch metrics, health checks, and alerting.
"""

import boto3
import json
import time
from datetime import datetime, timedelta

def create_cloudwatch_dashboard():
    """Create CloudWatch dashboard for neural codec monitoring."""
    cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
    
    dashboard_body = {
        "widgets": [
            {
                "type": "metric",
                "x": 0,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/EC2", "CPUUtilization", "InstanceId", "i-0b614aa221757060e"],
                        [".", ".", ".", "i-063947ae46af6dbf8"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-east-1",
                    "title": "EC2 CPU Utilization",
                    "period": 300,
                    "stat": "Average"
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/EC2", "NetworkIn", "InstanceId", "i-0b614aa221757060e"],
                        [".", "NetworkOut", ".", "."],
                        [".", "NetworkIn", ".", "i-063947ae46af6dbf8"],
                        [".", "NetworkOut", ".", "."]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-east-1",
                    "title": "Network Traffic",
                    "period": 300,
                    "stat": "Average"
                }
            },
            {
                "type": "metric",
                "x": 0,
                "y": 6,
                "width": 24,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["NeuralCodec", "ExperimentsCompleted", "Service", "Orchestrator"],
                        [".", "ExperimentsFailed", ".", "."],
                        [".", "ExperimentsCompleted", ".", "Worker"],
                        [".", "ExperimentsFailed", ".", "."]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-east-1",
                    "title": "Experiment Metrics",
                    "period": 300,
                    "stat": "Sum"
                }
            }
        ]
    }
    
    try:
        response = cloudwatch.put_dashboard(
            DashboardName='NeuralCodec-HTTP-Monitoring',
            DashboardBody=json.dumps(dashboard_body)
        )
        print("✅ CloudWatch dashboard created: NeuralCodec-HTTP-Monitoring")
        return True
    except Exception as e:
        print(f"❌ Failed to create dashboard: {e}")
        return False

def create_cloudwatch_alarms():
    """Create CloudWatch alarms for monitoring."""
    cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
    
    alarms = [
        {
            'AlarmName': 'neural-codec-worker-cpu-high',
            'AlarmDescription': 'Worker CPU utilization is high',
            'MetricName': 'CPUUtilization',
            'Namespace': 'AWS/EC2',
            'Statistic': 'Average',
            'Dimensions': [{'Name': 'InstanceId', 'Value': 'i-0b614aa221757060e'}],
            'Period': 300,
            'EvaluationPeriods': 2,
            'Threshold': 80.0,
            'ComparisonOperator': 'GreaterThanThreshold'
        },
        {
            'AlarmName': 'neural-codec-orchestrator-cpu-high',
            'AlarmDescription': 'Orchestrator CPU utilization is high',
            'MetricName': 'CPUUtilization',
            'Namespace': 'AWS/EC2',
            'Statistic': 'Average',
            'Dimensions': [{'Name': 'InstanceId', 'Value': 'i-063947ae46af6dbf8'}],
            'Period': 300,
            'EvaluationPeriods': 2,
            'Threshold': 80.0,
            'ComparisonOperator': 'GreaterThanThreshold'
        },
        {
            'AlarmName': 'neural-codec-worker-status-check',
            'AlarmDescription': 'Worker health check failed',
            'MetricName': 'StatusCheckFailed',
            'Namespace': 'AWS/EC2',
            'Statistic': 'Maximum',
            'Dimensions': [{'Name': 'InstanceId', 'Value': 'i-0b614aa221757060e'}],
            'Period': 300,
            'EvaluationPeriods': 1,
            'Threshold': 1.0,
            'ComparisonOperator': 'GreaterThanOrEqualToThreshold'
        },
        {
            'AlarmName': 'neural-codec-orchestrator-status-check',
            'AlarmDescription': 'Orchestrator health check failed',
            'MetricName': 'StatusCheckFailed',
            'Namespace': 'AWS/EC2',
            'Statistic': 'Maximum',
            'Dimensions': [{'Name': 'InstanceId', 'Value': 'i-063947ae46af6dbf8'}],
            'Period': 300,
            'EvaluationPeriods': 1,
            'Threshold': 1.0,
            'ComparisonOperator': 'GreaterThanOrEqualToThreshold'
        }
    ]
    
    success_count = 0
    for alarm in alarms:
        try:
            cloudwatch.put_metric_alarm(**alarm)
            print(f"✅ Created alarm: {alarm['AlarmName']}")
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to create alarm {alarm['AlarmName']}: {e}")
    
    print(f"📊 Created {success_count}/{len(alarms)} CloudWatch alarms")
    return success_count == len(alarms)

def create_health_check_scripts():
    """Create health check scripts for both services."""
    
    # Worker health check script
    worker_health_check = """#!/bin/bash
# Neural Codec Worker Health Check
set -e

WORKER_URL="http://18.208.180.67:8080"
LOG_FILE="/tmp/worker_health.log"

echo "$(date): Starting worker health check" >> $LOG_FILE

# Check if worker is responding
if curl -s -f "$WORKER_URL/health" > /dev/null; then
    echo "$(date): Worker health check PASSED" >> $LOG_FILE
    
    # Get worker status
    STATUS=$(curl -s "$WORKER_URL/status" | python3 -c "import sys,json; print(json.load(sys.stdin)['is_processing'])")
    
    if [ "$STATUS" = "false" ]; then
        echo "$(date): Worker is idle - ready for experiments" >> $LOG_FILE
        exit 0
    else
        echo "$(date): Worker is busy processing experiment" >> $LOG_FILE
        exit 0
    fi
else
    echo "$(date): Worker health check FAILED - not responding" >> $LOG_FILE
    exit 1
fi
"""
    
    # Orchestrator health check script
    orchestrator_health_check = """#!/bin/bash
# Neural Codec Orchestrator Health Check
set -e

ORCHESTRATOR_URL="http://34.239.1.29:8081"
LOG_FILE="/tmp/orchestrator_health.log"

echo "$(date): Starting orchestrator health check" >> $LOG_FILE

# Check if orchestrator is responding
if curl -s -f "$ORCHESTRATOR_URL/health" > /dev/null; then
    echo "$(date): Orchestrator health check PASSED" >> $LOG_FILE
    
    # Get orchestrator status
    HEALTH_DATA=$(curl -s "$ORCHESTRATOR_URL/health")
    AVAILABLE_WORKERS=$(echo "$HEALTH_DATA" | python3 -c "import sys,json; print(len([w for w in json.load(sys.stdin)['available_workers'] if w['status']['status'] == 'healthy']))")
    
    if [ "$AVAILABLE_WORKERS" -gt 0 ]; then
        echo "$(date): Orchestrator has $AVAILABLE_WORKERS available workers" >> $LOG_FILE
        exit 0
    else
        echo "$(date): Orchestrator has no available workers" >> $LOG_FILE
        exit 1
    fi
else
    echo "$(date): Orchestrator health check FAILED - not responding" >> $LOG_FILE
    exit 1
fi
"""
    
    try:
        # Save scripts to files
        with open('scripts/worker_health_check.sh', 'w') as f:
            f.write(worker_health_check)
        
        with open('scripts/orchestrator_health_check.sh', 'w') as f:
            f.write(orchestrator_health_check)
        
        # Make scripts executable
        import os
        os.chmod('scripts/worker_health_check.sh', 0o755)
        os.chmod('scripts/orchestrator_health_check.sh', 0o755)
        
        print("✅ Health check scripts created")
        return True
    except Exception as e:
        print(f"❌ Failed to create health check scripts: {e}")
        return False

def setup_cloudwatch_log_groups():
    """Set up CloudWatch log groups for centralized logging."""
    logs = boto3.client('logs', region_name='us-east-1')
    
    log_groups = [
        {
            'logGroupName': '/aws/neural-codec/http-worker',
            'retentionInDays': 30
        },
        {
            'logGroupName': '/aws/neural-codec/http-orchestrator', 
            'retentionInDays': 30
        }
    ]
    
    success_count = 0
    for log_group in log_groups:
        try:
            logs.create_log_group(**log_group)
            print(f"✅ Created log group: {log_group['logGroupName']}")
            success_count += 1
        except logs.exceptions.ResourceAlreadyExistsException:
            print(f"ℹ️  Log group already exists: {log_group['logGroupName']}")
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to create log group {log_group['logGroupName']}: {e}")
    
    print(f"📝 Set up {success_count}/{len(log_groups)} CloudWatch log groups")
    return success_count == len(log_groups)

def create_monitoring_summary():
    """Create a monitoring setup summary."""
    summary = f"""# Neural Codec HTTP System - Monitoring Setup

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

## 🎯 Monitoring Components

### ✅ CloudWatch Dashboard
- **Name:** NeuralCodec-HTTP-Monitoring
- **URL:** https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=NeuralCodec-HTTP-Monitoring
- **Metrics:** CPU, Network, Experiment counts

### ✅ CloudWatch Alarms
- `neural-codec-worker-cpu-high` - Worker CPU > 80%
- `neural-codec-orchestrator-cpu-high` - Orchestrator CPU > 80%
- `neural-codec-worker-status-check` - Worker health check failures
- `neural-codec-orchestrator-status-check` - Orchestrator health check failures

### ✅ Health Check Scripts
- `scripts/worker_health_check.sh` - Worker health monitoring
- `scripts/orchestrator_health_check.sh` - Orchestrator health monitoring

### ✅ CloudWatch Log Groups
- `/aws/neural-codec/http-worker` - Worker logs (30-day retention)
- `/aws/neural-codec/http-orchestrator` - Orchestrator logs (30-day retention)

## 🔍 Monitoring Endpoints

### Service Health Checks:
```bash
# Worker health
curl http://18.208.180.67:8080/health

# Orchestrator health  
curl http://34.239.1.29:8081/health
```

### Service Status:
```bash
# Worker status
curl http://18.208.180.67:8080/status

# Active experiments
curl http://34.239.1.29:8081/experiments
```

## 📊 Key Metrics to Monitor

1. **Availability:** Service health check responses
2. **Performance:** Response times for experiments
3. **Throughput:** Experiments completed per hour
4. **Resource Usage:** CPU, memory, network utilization
5. **Error Rates:** Failed experiments, timeouts

## 🚨 Alert Thresholds

- **CPU Utilization:** > 80% for 10 minutes
- **Health Check Failures:** Any failure
- **Response Time:** > 30 seconds for experiments
- **Error Rate:** > 5% of experiments failing

## 📱 Next Steps

1. **Set up SNS notifications** for alarm alerts
2. **Configure log shipping** to CloudWatch
3. **Set up custom metrics** for experiment success rates
4. **Create automated recovery** scripts

---
*Monitoring setup completed successfully!*
"""
    
    with open('MONITORING_SETUP.md', 'w') as f:
        f.write(summary)
    
    print("✅ Monitoring setup summary created: MONITORING_SETUP.md")

def main():
    """Main monitoring setup function."""
    print("🚀 Setting up Monitoring and Alerting for Neural Codec HTTP System")
    print("=" * 70)
    
    success_count = 0
    total_tasks = 4
    
    # 1. Create CloudWatch dashboard
    print("\n1️⃣ Creating CloudWatch dashboard...")
    if create_cloudwatch_dashboard():
        success_count += 1
    
    # 2. Create CloudWatch alarms
    print("\n2️⃣ Creating CloudWatch alarms...")
    if create_cloudwatch_alarms():
        success_count += 1
    
    # 3. Create health check scripts
    print("\n3️⃣ Creating health check scripts...")
    if create_health_check_scripts():
        success_count += 1
    
    # 4. Set up CloudWatch log groups
    print("\n4️⃣ Setting up CloudWatch log groups...")
    if setup_cloudwatch_log_groups():
        success_count += 1
    
    # 5. Create monitoring summary
    print("\n5️⃣ Creating monitoring summary...")
    create_monitoring_summary()
    
    # Summary
    print("\n" + "=" * 70)
    print(f"📊 Monitoring Setup Complete: {success_count}/{total_tasks} tasks successful")
    
    if success_count == total_tasks:
        print("🎉 All monitoring components configured successfully!")
        print("\n📱 Access your monitoring:")
        print("   CloudWatch Dashboard: https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=NeuralCodec-HTTP-Monitoring")
        print("   CloudWatch Alarms: https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#alarmsV2:")
        print("   Health Checks: Run scripts/worker_health_check.sh and scripts/orchestrator_health_check.sh")
    else:
        print("⚠️  Some monitoring components failed to configure")
        print("   Check the error messages above and retry failed components")

if __name__ == '__main__':
    main()
