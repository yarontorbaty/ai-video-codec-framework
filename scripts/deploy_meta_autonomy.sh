#!/bin/bash
# Deploy meta-autonomy features (tool calling for framework modification)

set -e

echo "🚀 Deploying Meta-Autonomy Features..."
echo ""

# Configuration
INSTANCE_NAME="ai-video-codec-orchestrator"
REGION="us-east-1"

# Get orchestrator instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text \
    --region $REGION)

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" == "None" ]; then
    echo "❌ Orchestrator instance not found or not running"
    exit 1
fi

echo "✅ Found orchestrator: $INSTANCE_ID"
echo ""

# Create deployment package
echo "📦 Packaging files..."
cd "$(dirname "$0")/.."

# Create temp directory
TEMP_DIR=$(mktemp -d)
mkdir -p "$TEMP_DIR/src/utils"
mkdir -p "$TEMP_DIR/src/agents"

# Copy new files
cp src/utils/framework_modifier.py "$TEMP_DIR/src/utils/"
cp src/agents/llm_experiment_planner.py "$TEMP_DIR/src/agents/"
cp LLM_SYSTEM_PROMPT.md "$TEMP_DIR/"

# Create tarball
cd "$TEMP_DIR"
tar -czf /tmp/meta_autonomy.tar.gz .
cd - > /dev/null

echo "✅ Package created: /tmp/meta_autonomy.tar.gz"
echo ""

# Upload to S3
echo "📤 Uploading to S3..."
aws s3 cp /tmp/meta_autonomy.tar.gz s3://ai-video-codec-artifacts-580473065386/deployments/ --region $REGION

echo "✅ Uploaded to S3"
echo ""

# Deploy to orchestrator via SSM
echo "🚀 Deploying to orchestrator..."

COMMAND_ID=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "cd /home/ec2-user",
        "aws s3 cp s3://ai-video-codec-artifacts-580473065386/deployments/meta_autonomy.tar.gz /tmp/",
        "cd /home/ec2-user/ai-video-codec",
        "tar -xzf /tmp/meta_autonomy.tar.gz",
        "echo \"✅ Files deployed\"",
        "echo \"🔄 Restarting orchestrator...\"",
        "pkill -f autonomous_orchestrator_llm.sh || true",
        "sleep 2",
        "nohup bash scripts/autonomous_orchestrator_llm.sh > /tmp/orch.log 2>&1 &",
        "sleep 3",
        "pgrep -f autonomous_orchestrator_llm.sh && echo \"✅ Orchestrator running (PID: $(pgrep -f autonomous_orchestrator_llm.sh))\" || echo \"❌ Orchestrator failed to start\""
    ]' \
    --region $REGION \
    --query 'Command.CommandId' \
    --output text)

echo "Command ID: $COMMAND_ID"
echo ""
echo "⏳ Waiting for deployment (30s)..."
sleep 30

# Check result
echo ""
echo "📊 Deployment Output:"
aws ssm get-command-invocation \
    --command-id "$COMMAND_ID" \
    --instance-id "$INSTANCE_ID" \
    --region $REGION \
    --query 'StandardOutputContent' \
    --output text

echo ""
echo "🎉 META-AUTONOMY DEPLOYED!"
echo ""
echo "The LLM can now:"
echo "  • Modify framework files"
echo "  • Run shell commands"
echo "  • Install packages"
echo "  • Restart orchestrator"
echo "  • Rollback changes"
echo ""
echo "Monitor: ssh to orchestrator and check /tmp/orch.log"

# Cleanup
rm -f /tmp/meta_autonomy.tar.gz
rm -rf "$TEMP_DIR"

