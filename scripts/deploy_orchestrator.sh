#!/bin/bash
# Deploy orchestrator with all dependencies

set -e

INSTANCE_ID="i-063947ae46af6dbf8"
REGION="us-east-1"
S3_BUCKET="ai-video-codec-artifacts-580473065386"

echo "============================================================"
echo "DEPLOYING ORCHESTRATOR WITH DEPENDENCIES"
echo "============================================================"
echo ""
echo "Instance: $INSTANCE_ID"
echo "Region: $REGION"
echo ""

# Step 1: Package code
echo "📦 Step 1: Packaging code..."
cd "$(dirname "$0")/.."
tar czf /tmp/orchestrator_deploy.tar.gz \
    src/ \
    scripts/real_experiment.py \
    scripts/autonomous_orchestrator_llm.sh \
    scripts/analyze_past_experiments.py \
    LLM_SYSTEM_PROMPT.md \
    NEURAL_NETWORKS_GUIDE.md \
    requirements.txt 2>/dev/null || true

echo "✅ Code packaged (including system prompt)"
echo ""

# Step 2: Upload to S3
echo "📤 Step 2: Uploading to S3..."
aws s3 cp /tmp/orchestrator_deploy.tar.gz \
    s3://$S3_BUCKET/deployment/orchestrator_latest.tar.gz

echo "✅ Uploaded to S3"
echo ""

# Step 3: Deploy to EC2
echo "🚀 Step 3: Deploying to EC2..."

COMMAND_ID=$(aws ssm send-command \
    --instance-ids $INSTANCE_ID \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "set -e",
        "echo \"=== Stopping orchestrator ===\"",
        "pkill -f autonomous_orchestrator_llm.sh || true",
        "sleep 2",
        "",
        "echo \"=== Downloading code ===\"",
        "cd /home/ec2-user/ai-video-codec",
        "aws s3 cp s3://ai-video-codec-artifacts-580473065386/deployment/orchestrator_latest.tar.gz .",
        "tar xzf orchestrator_latest.tar.gz",
        "rm orchestrator_latest.tar.gz",
        "",
        "echo \"=== Installing Python dependencies ===\"",
        "pip3 install --upgrade boto3 numpy opencv-python-headless torch torchvision 2>&1 | tail -20",
        "",
        "echo \"=== Trying to install anthropic (may fail on Python 3.7) ===\"",
        "pip3 install anthropic 2>&1 | tail -10 || echo \"Anthropic install failed - will use fallback API\"",
        "",
        "echo \"=== Verifying API key is accessible ===\"",
        "SECRET_CHECK=$(aws secretsmanager get-secret-value --secret-id ai-video-codec/anthropic-api-key --region us-east-1 --query SecretString --output text 2>&1 | head -c 50)",
        "if [[ \"$SECRET_CHECK\" == *ANTHROPIC_API_KEY* ]]; then",
        "    echo \"✅ API key found in Secrets Manager (will be loaded by orchestrator)\"",
        "else",
        "    echo \"❌ API key not found in Secrets Manager\"",
        "    echo \"Check: $SECRET_CHECK\"",
        "fi",
        "",
        "echo \"=== Starting orchestrator ===\"",
        "cd /home/ec2-user/ai-video-codec",
        "nohup bash scripts/autonomous_orchestrator_llm.sh > /tmp/orch.log 2>&1 &",
        "sleep 3",
        "",
        "echo \"=== Verifying orchestrator started ===\"",
        "ps aux | grep autonomous_orchestrator | grep -v grep",
        "NEW_PID=$(pgrep -f autonomous_orchestrator_llm.sh)",
        "echo \"✅ Orchestrator running (PID: $NEW_PID)\"",
        "",
        "echo \"\"",
        "echo \"============================================================\"",
        "echo \"✅ DEPLOYMENT COMPLETE\"",
        "echo \"============================================================\"",
        "echo \"\"",
        "echo \"Orchestrator logs: tail -f /tmp/orch.log\"",
        "echo \"\"",
        "exit 0"
    ]' \
    --region $REGION \
    --query 'Command.CommandId' \
    --output text)

echo "Command ID: $COMMAND_ID"
echo ""
echo "⏳ Waiting for deployment to complete..."
sleep 20

# Step 4: Get results
echo ""
echo "📋 Deployment output:"
echo "============================================================"
aws ssm get-command-invocation \
    --command-id $COMMAND_ID \
    --instance-id $INSTANCE_ID \
    --region $REGION \
    --query 'StandardOutputContent' \
    --output text

echo ""
echo "============================================================"
echo "✅ DEPLOYMENT SCRIPT COMPLETE"
echo "============================================================"
echo ""
echo "To check orchestrator status:"
echo "  aws ssm send-command --instance-ids $INSTANCE_ID \\"
echo "    --document-name \"AWS-RunShellScript\" \\"
echo "    --parameters 'commands=[\"tail -50 /tmp/orch.log\"]' \\"
echo "    --query 'Command.CommandId' --output text"
echo ""

