#!/bin/bash
#
# Deploy v2.0 via AWS SSM
#

set -e

echo "=========================================="
echo "🚀 DEPLOYING v2.0 VIA AWS SSM"
echo "=========================================="
echo ""

# Instance IDs
ORCHESTRATOR="i-063947ae46af6dbf8"
GPU_WORKER="i-0b614aa221757060e"  # g4dn.xlarge inference worker

# Step 1: Deploy to Orchestrator
echo "📦 Deploying to Orchestrator ($ORCHESTRATOR)..."

aws ssm send-command \
    --instance-ids "$ORCHESTRATOR" \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "cd /home/ubuntu/ai-video-codec-framework || exit 1",
        "git fetch origin || true",
        "source venv/bin/activate || python3 -m venv venv && source venv/bin/activate",
        "pip3 install -q scikit-image thop boto3 anthropic",
        "echo \"✅ Orchestrator dependencies updated\""
    ]' \
    --output text \
    --query 'Command.CommandId'

echo "✅ Orchestrator deployment initiated"
echo ""

# Step 2: Deploy to GPU Worker
echo "📦 Deploying to GPU Worker ($GPU_WORKER)..."

aws ssm send-command \
    --instance-ids "$GPU_WORKER" \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "cd /home/ubuntu/ai-video-codec-framework || exit 1",
        "git fetch origin || true",
        "source venv/bin/activate || python3 -m venv venv && source venv/bin/activate",
        "pip3 install -q scikit-image thop",
        "python3 -c \"import torch; print(f\"CUDA: {torch.cuda.is_available()}\")",
        "echo \"✅ GPU Worker dependencies updated\""
    ]' \
    --output text \
    --query 'Command.CommandId'

echo "✅ GPU Worker deployment initiated"
echo ""

echo "=========================================="
echo "✅ DEPLOYMENT COMMANDS SENT"
echo "=========================================="
echo ""
echo "Waiting 10 seconds for commands to complete..."
sleep 10

echo ""
echo "To check status:"
echo "  aws ssm list-command-invocations --instance-id $ORCHESTRATOR --details"
echo ""

