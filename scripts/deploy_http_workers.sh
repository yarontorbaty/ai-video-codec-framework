#!/bin/bash
set -e

echo "🚀 Deploying HTTP-based Neural Codec Workers"
echo "=============================================="

# Configuration
WORKER_INSTANCE="i-0b614aa221757060e"
ORCHESTRATOR_INSTANCE="i-063947ae46af6dbf8"
WORKER_IP=$(aws ec2 describe-instances --instance-ids $WORKER_INSTANCE --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
ORCHESTRATOR_IP=$(aws ec2 describe-instances --instance-ids $ORCHESTRATOR_INSTANCE --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "📍 Instance Details:"
echo "   Worker Instance: $WORKER_INSTANCE ($WORKER_IP)"
echo "   Orchestrator Instance: $ORCHESTRATOR_INSTANCE ($ORCHESTRATOR_IP)"

# Upload HTTP worker to S3
echo ""
echo "📤 Uploading HTTP worker to S3..."
aws s3 cp workers/neural_codec_http_worker.py s3://ai-video-codec-deployment/workers/neural_codec_http_worker.py
aws s3 cp src/agents/http_orchestrator.py s3://ai-video-codec-deployment/agents/http_orchestrator.py
aws s3 cp scripts/test_http_pipeline.py s3://ai-video-codec-deployment/scripts/test_http_pipeline.py

# Deploy HTTP worker
echo ""
echo "🔧 Deploying HTTP worker to GPU instance..."
aws ssm send-command \
    --instance-ids $WORKER_INSTANCE \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "cd /home/ubuntu/ai-video-codec-framework",
        "aws s3 cp s3://ai-video-codec-deployment/workers/neural_codec_http_worker.py workers/",
        "aws s3 cp s3://ai-video-codec-deployment/scripts/test_http_pipeline.py scripts/",
        "pip3 install flask requests",
        "pkill -f neural_codec_gpu_worker || true",
        "pkill -f neural_codec_http_worker || true",
        "sleep 2",
        "nohup python3 workers/neural_codec_http_worker.py > /tmp/http_worker.log 2>&1 &",
        "sleep 3",
        "ps aux | grep neural_codec_http_worker | grep -v grep || echo \"Worker not running\"",
        "echo \"=== HTTP Worker Logs ===\"",
        "tail -10 /tmp/http_worker.log"
    ]' \
    --output text --query 'Command.CommandId'

# Deploy HTTP orchestrator
echo ""
echo "🔧 Deploying HTTP orchestrator..."
aws ssm send-command \
    --instance-ids $ORCHESTRATOR_INSTANCE \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "cd /home/ubuntu/ai-video-codec-framework",
        "aws s3 cp s3://ai-video-codec-deployment/agents/http_orchestrator.py src/agents/",
        "pip3 install flask requests",
        "pkill -f gpu_first_orchestrator || true",
        "pkill -f http_orchestrator || true",
        "sleep 2",
        "export WORKER_URLS=\"http://'$WORKER_IP':8080\"",
        "nohup python3 src/agents/http_orchestrator.py > /tmp/http_orchestrator.log 2>&1 &",
        "sleep 3",
        "ps aux | grep http_orchestrator | grep -v grep || echo \"Orchestrator not running\"",
        "echo \"=== HTTP Orchestrator Logs ===\"",
        "tail -10 /tmp/http_orchestrator.log"
    ]' \
    --output text --query 'Command.CommandId'

echo ""
echo "⏳ Waiting for deployment to complete..."
sleep 15

echo ""
echo "🔍 Checking deployment status..."

# Check worker status
echo "Worker Status:"
aws ssm send-command \
    --instance-ids $WORKER_INSTANCE \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "curl -s http://localhost:8080/health | python3 -m json.tool || echo \"Worker not responding\"",
        "echo \"=== Worker Logs ===\"",
        "tail -20 /tmp/http_worker.log"
    ]' \
    --output text --query 'Command.CommandId'

# Check orchestrator status
echo "Orchestrator Status:"
aws ssm send-command \
    --instance-ids $ORCHESTRATOR_INSTANCE \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "curl -s http://localhost:8081/health | python3 -m json.tool || echo \"Orchestrator not responding\"",
        "echo \"=== Orchestrator Logs ===\"",
        "tail -20 /tmp/http_orchestrator.log"
    ]' \
    --output text --query 'Command.CommandId'

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Service URLs:"
echo "   Worker: http://$WORKER_IP:8080"
echo "   Orchestrator: http://$ORCHESTRATOR_IP:8081"
echo ""
echo "🧪 To test the pipeline:"
echo "   python3 scripts/test_http_pipeline.py"
echo ""
echo "📊 To check logs:"
echo "   Worker: ssh ubuntu@$WORKER_IP 'tail -f /tmp/http_worker.log'"
echo "   Orchestrator: ssh ubuntu@$ORCHESTRATOR_IP 'tail -f /tmp/http_orchestrator.log'"
