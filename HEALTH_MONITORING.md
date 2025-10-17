# Health Monitoring and Auto-Healing System

## Overview

The AI Video Codec orchestrator now has a comprehensive health monitoring and auto-healing system that ensures continuous operation with minimal manual intervention.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   EventBridge Schedule                       │
│              (Triggers every 5 minutes)                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            Health Monitor Lambda Function                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  1. Check orchestrator process running                │  │
│  │  2. Verify recent experiments (last 10 min)           │  │
│  │  3. Monitor system resources (CPU, Memory, Disk)      │  │
│  │  4. Auto-heal if unhealthy                            │  │
│  │  5. Send metrics to CloudWatch                        │  │
│  │  6. Log status to DynamoDB                            │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│  CloudWatch  │ │ DynamoDB │ │ Auto-Healing │
│   Metrics    │ │  Control │ │   Actions    │
│  & Alarms    │ │   Table  │ │              │
└──────────────┘ └──────────┘ └──────────────┘
```

## Health Checks

The system performs the following health checks every 5 minutes:

### 1. Process Health
- **Check**: Is the orchestrator script running?
- **Command**: `ps aux | grep autonomous_orchestrator_llm.sh`
- **Action if Failed**: Restart orchestrator

### 2. Activity Health
- **Check**: Have experiments run in the last 10 minutes?
- **Source**: DynamoDB experiments table
- **Action if Failed**: Restart orchestrator (likely stuck)

### 3. Resource Health
- **CPU Usage**: Warning if > 80%, critical if > 95%
- **Memory Usage**: Warning if > 90%, critical if > 95%
- **Disk Usage**: Warning if > 85%, critical if > 90%
- **Action if Failed**: Clean up disk space, then restart if needed

## Auto-Healing Actions

When the system detects unhealthy conditions, it automatically:

### Disk Cleanup
```bash
- Remove temporary files older than 7 days
- Vacuum systemd journal logs
- Remove old video files (*.mp4)
```

### Orchestrator Restart
```bash
1. Kill existing orchestrator process
2. Wait 2 seconds
3. Start new orchestrator process
4. Verify it's running
```

## CloudWatch Metrics

The following custom metrics are published every 5 minutes:

| Metric Name | Description | Healthy Value |
|-------------|-------------|---------------|
| `OrchestratorRunning` | Process is running | 1.0 |
| `RecentExperiments` | Experiments in last 10 min | 1.0 |
| `OverallHealth` | All checks passing | 1.0 |
| `CPUUsage` | CPU utilization % | < 80% |
| `MemoryUsage` | Memory utilization % | < 90% |
| `DiskUsage` | Disk utilization % | < 85% |

**Namespace**: `AiVideoCodec/Orchestrator`

## CloudWatch Alarms

### 1. Orchestrator Down
- **Condition**: OrchestratorRunning < 0.5 for 10 minutes (2 periods)
- **Action**: SNS alert

### 2. No Recent Experiments
- **Condition**: RecentExperiments < 0.5 for 10 minutes
- **Action**: SNS alert

### 3. High Disk Usage
- **Condition**: DiskUsage > 85% for 10 minutes (2 periods)
- **Action**: SNS alert + automatic cleanup

### 4. High Memory Usage
- **Condition**: MemoryUsage > 90% for 10 minutes (2 periods)
- **Action**: SNS alert

## Monitoring Dashboard

View real-time health metrics at:
```
https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=ai-video-codec-health-monitoring
```

The dashboard shows:
- Orchestrator health status (running, experiments, overall)
- Resource usage trends (CPU, memory, disk)
- Historical health data

## Manual Health Check

To manually trigger a health check:

```bash
aws lambda invoke \
  --function-name ai-video-codec-health-monitor \
  /tmp/health_check.json

cat /tmp/health_check.json | jq .
```

## Viewing Health Logs

### Lambda Logs
```bash
aws logs tail /aws/lambda/ai-video-codec-health-monitor --follow
```

### Orchestrator Logs
```bash
aws ssm send-command \
  --instance-ids i-063947ae46af6dbf8 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["tail -50 /tmp/orch.log"]'
```

### DynamoDB Health Status
```bash
aws dynamodb query \
  --table-name ai-video-codec-control \
  --index-name type-index \
  --key-condition-expression "#type = :type" \
  --expression-attribute-names '{"#type":"type"}' \
  --expression-attribute-values '{":type":{"S":"health_check"}}' \
  --limit 10 \
  --scan-index-forward false
```

## Configuration

### Adjust Health Check Frequency

Edit the CloudFormation parameter:
```yaml
HealthCheckSchedule: rate(5 minutes)  # Change to rate(1 minute) for more frequent checks
```

Then update the stack:
```bash
aws cloudformation update-stack \
  --stack-name ai-video-codec-health-monitoring \
  --use-previous-template \
  --parameters ParameterKey=HealthCheckSchedule,ParameterValue="rate(1 minute)"
```

### Configure SNS Alerts

Subscribe to the SNS topic for email/SMS alerts:
```bash
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:580473065386:ai-video-codec-health-alerts \
  --protocol email \
  --notification-endpoint your-email@example.com
```

## Troubleshooting

### Health Monitor Not Running

Check EventBridge rule is enabled:
```bash
aws events describe-rule --name ai-video-codec-health-check-schedule
```

### Auto-Healing Not Working

Check Lambda execution role permissions:
```bash
aws iam get-role-policy \
  --role-name ai-video-codec-health-monitor-role \
  --policy-name HealthMonitorPermissions
```

Ensure the role has:
- `ssm:SendCommand` - To run commands on orchestrator
- `ssm:GetCommandInvocation` - To check command results
- `dynamodb:*` - To read experiments and log health status
- `cloudwatch:PutMetricData` - To send metrics

### Orchestrator Keeps Restarting

Check the orchestrator logs for errors:
```bash
aws ssm send-command \
  --instance-ids i-063947ae46af6dbf8 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["tail -100 /tmp/orch.log | grep -i error"]'
```

Common issues:
1. **Missing dependencies**: Run `pip3 install --user boto3 anthropic opencv-python-headless numpy`
2. **API key not set**: Check Secrets Manager has the Anthropic API key
3. **Insufficient permissions**: Check EC2 instance IAM role

## Cost Estimate

The health monitoring system costs approximately:

- **Lambda invocations**: 8,640 per month (every 5 min) = $0.00
- **Lambda duration**: ~5 seconds per check = $0.01
- **CloudWatch metrics**: 6 custom metrics = $0.30/month
- **CloudWatch alarms**: 4 alarms = $0.40/month
- **CloudWatch dashboard**: 1 dashboard = $3.00/month
- **SNS notifications**: < 1,000 per month = $0.00
- **DynamoDB writes**: ~8,640 per month = $0.01

**Total**: ~$3.72/month

## Benefits

✅ **Zero downtime** - Automatic recovery from failures  
✅ **Proactive monitoring** - Detect issues before they cause problems  
✅ **Historical tracking** - All health data logged to DynamoDB  
✅ **Alert integration** - SNS notifications for critical issues  
✅ **Resource optimization** - Automatic cleanup of disk space  
✅ **Cost efficient** - < $4/month for 24/7 monitoring  

## Next Steps

1. **Subscribe to SNS alerts**: Add your email to get notified of issues
2. **Review dashboard daily**: Check CloudWatch dashboard for trends
3. **Tune thresholds**: Adjust alarm thresholds based on actual usage
4. **Add custom checks**: Extend health_monitor.py with project-specific checks

