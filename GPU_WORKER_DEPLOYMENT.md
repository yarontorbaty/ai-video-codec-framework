# GPU Worker Deployment Guide

## Overview

Your GPU instances are ready to execute PyTorch-based experiments! The code is committed and ready - you just need to deploy the worker script to each GPU instance.

## Current Status

✅ **GPU Instances Running:**
- 2× `g5.4xlarge` (NVIDIA A10G, 24GB VRAM) - Training workers
- 1× `g4dn.xlarge` (NVIDIA T4, 16GB VRAM) - Inference worker

✅ **Code Complete:**
- `workers/training_worker.py` - GPU worker with SQS polling
- `src/utils/code_sandbox.py` - GPU detection logic
- `src/agents/procedural_experiment_runner.py` - Automatic GPU dispatching

✅ **Infrastructure:**
- SQS queue: `ai-video-codec-training-queue`
- DynamoDB integration for results
- Automatic PyTorch code detection

⚠️ **SSM Agent Not Configured:**
- GPU instances don't have SSM agent enabled
- Manual deployment required (or SSH if you have key pairs)

## Manual Deployment (via SSH)

If you have SSH access to the GPU instances:

```bash
# SSH into each GPU instance
ssh -i your-key.pem ec2-user@<instance-ip>

# Run deployment script
cd /tmp
cat > deploy_worker.sh <<'EOF'
#!/bin/bash
set -e

echo "Setting up GPU Training Worker..."

# Setup directory
sudo mkdir -p /opt/ai-video-codec
sudo chown ec2-user:ec2-user /opt/ai-video-codec
cd /opt/ai-video-codec

# Clone repository
if [ -d ".git" ]; then
    git fetch origin
    git reset --hard origin/main
else
    git clone https://github.com/yarontorbaty/ai-video-codec-framework.git .
fi

# Install PyTorch with CUDA support
pip3 install --user torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install --user boto3

# Create systemd service
sudo tee /etc/systemd/system/ai-video-codec-worker.service > /dev/null <<'SERVICE'
[Unit]
Description=AI Video Codec GPU Training Worker
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/ai-video-codec
Environment="TRAINING_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/580473065386/ai-video-codec-training-queue"
Environment="PYTHONPATH=/opt/ai-video-codec"
ExecStart=/usr/bin/python3 /opt/ai-video-codec/workers/training_worker.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE

# Start worker
sudo systemctl daemon-reload
sudo systemctl enable ai-video-codec-worker
sudo systemctl restart ai-video-codec-worker

# Check status
sleep 2
sudo systemctl status ai-video-codec-worker --no-pager
EOF

chmod +x deploy_worker.sh
./deploy_worker.sh
```

## Alternative: Enable SSM Agent

If you prefer automated deployment via SSM:

```bash
# SSH into each GPU instance and install SSM agent
sudo yum install -y amazon-ssm-agent
sudo systemctl enable amazon-ssm-agent
sudo systemctl start amazon-ssm-agent
```

Then use the automated deployment script:

```bash
cd /Users/yarontorbaty/Documents/Code/AiV1
bash scripts/deploy_gpu_workers.sh
```

## How It Works

### 1. **LLM Generates PyTorch Code**
```python
# Example LLM output
import torch
import torch.nn as nn

def compress_video_frame(frame, frame_index, config):
    model = MyNeuralCodec()
    compressed = model.encode(frame)
    return compressed.cpu().numpy().tobytes()
```

### 2. **Orchestrator Detects GPU Requirement**
```python
# In procedural_experiment_runner.py
requires_gpu = CodeSandbox.requires_gpu(code)  # Returns True for PyTorch code
```

### 3. **Job Dispatched to SQS**
```python
sqs.send_message(
    QueueUrl=TRAINING_QUEUE_URL,
    MessageBody=json.dumps({
        'experiment_id': 'proc_exp_1234',
        'code': code,
        'config': {...}
    })
)
```

### 4. **GPU Worker Executes**
```python
# GPU worker polls SQS
job = sqs.receive_message(QueueUrl=TRAINING_QUEUE_URL)

# Execute on GPU
results = executor.execute_experiment(job)

# Store results in DynamoDB
experiments_table.update_item(
    Key={'experiment_id': experiment_id},
    UpdateExpression='SET gpu_execution_results = :results'
)
```

### 5. **Orchestrator Receives Results**
- Polls DynamoDB for GPU execution results
- Continues with quality verification phase
- Displays in dashboard with "Execution Device: GPU" badge

## Monitoring

### Check Worker Status
```bash
# Via SSM (if configured)
aws ssm send-command --instance-ids i-08d0cb8a128aac0d6 \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["sudo systemctl status ai-video-codec-worker"]'

# Via SSH
ssh ec2-user@<instance-ip>
sudo systemctl status ai-video-codec-worker
sudo journalctl -u ai-video-codec-worker -f
```

### Check SQS Queue
```bash
# View queue stats
aws sqs get-queue-attributes \
    --queue-url https://sqs.us-east-1.amazonaws.com/580473065386/ai-video-codec-training-queue \
    --attribute-names All

# Check for messages
aws sqs receive-message \
    --queue-url https://sqs.us-east-1.amazonaws.com/580473065386/ai-video-codec-training-queue \
    --max-number-of-messages 1
```

## Cost Optimization

### Current GPU Costs (if running 24/7):
- 2× g5.4xlarge: $2.04/hr × 2 = $4.08/hr = **$2,938/month**
- 1× g4dn.xlarge: $0.526/hr = **$379/month**
- **Total: ~$3,317/month**

### Recommended: Use Spot Instances
Update CloudFormation template to use Spot pricing:

```yaml
TrainingWorkerLaunchTemplate:
  Properties:
    LaunchTemplateData:
      InstanceMarketOptions:
        MarketType: spot
        SpotOptions:
          MaxPrice: "1.50"  # ~70% savings
          SpotInstanceType: persistent
```

**Potential Savings: ~$2,322/month (70% reduction)**

### Auto-Scaling by Queue Depth

Add CloudWatch alarm to scale workers based on SQS queue depth:

```yaml
ScaleUpPolicy:
  Type: AWS::AutoScaling::ScalingPolicy
  Properties:
    AutoScalingGroupName: !Ref TrainingWorkerASG
    PolicyType: TargetTrackingScaling
    TargetTrackingConfiguration:
      CustomizedMetricSpecification:
        MetricName: ApproximateNumberOfMessagesVisible
        Namespace: AWS/SQS
        Statistic: Average
      TargetValue: 5  # Scale up when >5 messages in queue
```

## Testing

### Test GPU Detection
```python
from utils.code_sandbox import CodeSandbox

# Should return True
code_with_torch = """
import torch
def compress_video_frame(frame, frame_index, config):
    model = torch.nn.Linear(100, 10)
    return model(frame).detach().numpy().tobytes()
"""
print(CodeSandbox.requires_gpu(code_with_torch))  # True

# Should return False
code_without_torch = """
import cv2
def compress_video_frame(frame, frame_index, config):
    return cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])[1].tobytes()
"""
print(CodeSandbox.requires_gpu(code_without_torch))  # False
```

### Monitor First GPU Experiment
```bash
# Watch orchestrator logs
aws ssm send-command --instance-ids i-063947ae46af6dbf8 \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["tail -f /home/ubuntu/orchestrator.log | grep -i gpu"]'

# Watch GPU worker logs (once deployed)
ssh ec2-user@<gpu-instance-ip>
sudo journalctl -u ai-video-codec-worker -f
```

## Troubleshooting

### Worker Not Starting
```bash
# Check logs
sudo journalctl -u ai-video-codec-worker -n 50

# Common issues:
# 1. PyTorch not installed: pip3 install torch
# 2. boto3 not installed: pip3 install boto3
# 3. Wrong Python path: which python3
# 4. CUDA not available: nvidia-smi
```

### Jobs Not Being Processed
```bash
# 1. Check if worker is polling
sudo journalctl -u ai-video-codec-worker -f | grep -i polling

# 2. Check SQS permissions
aws sqs get-queue-attributes \
    --queue-url https://sqs.us-east-1.amazonaws.com/580473065386/ai-video-codec-training-queue

# 3. Manually send test message
aws sqs send-message \
    --queue-url https://sqs.us-east-1.amazonaws.com/580473065386/ai-video-codec-training-queue \
    --message-body '{"test": true}'
```

### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall CUDA PyTorch
pip3 uninstall torch torchvision
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify in Python
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Next Steps

1. **Deploy Workers** (choose one method above)
2. **Verify GPU Detection**: `nvidia-smi` on each instance
3. **Start Worker Services**: Workers will auto-poll SQS
4. **LLM Will Generate GPU Code**: Next experiments will use PyTorch
5. **Monitor Dashboard**: Look for "Execution Device: GPU" badge

## Expected Results

Once deployed, you'll see:
- ✅ GPU workers picking up PyTorch-based experiments
- ✅ Faster execution times (10-100x for neural networks)
- ✅ Dashboard showing "GPU" execution device
- ✅ Better compression results from neural codecs

The LLM is already configured to generate PyTorch code - it just needs working GPU infrastructure to execute it!

