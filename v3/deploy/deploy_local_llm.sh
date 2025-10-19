#!/bin/bash
# Deploy Local LLM System for Unlimited Codec Generation

set -e

echo "============================================================"
echo "🚀 Deploying Local LLM System"
echo "============================================================"
echo ""

# Configuration
REGION="us-east-1"
INSTANCE_TYPE="g4dn.xlarge"  # $0.526/hour, 1 T4 GPU, 16GB RAM
AMI_ID="ami-09e2639b59ee94f7c"  # Deep Learning AMI (PyTorch) - has CUDA pre-installed
KEY_NAME="gpu-ec2"
SECURITY_GROUP="sg-0e573e2f685e36cb9"
SUBNET_ID="subnet-2f7b1b4a"
IAM_ROLE="ai-codec-v3-ec2-role"

echo "📋 Configuration:"
echo "   Instance Type: $INSTANCE_TYPE"
echo "   Region: $REGION"
echo "   GPU: NVIDIA T4 (16GB)"
echo ""

# Launch GPU instance
echo "🚀 Launching GPU instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP \
    --subnet-id $SUBNET_ID \
    --iam-instance-profile Name=$IAM_ROLE \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ai-codec-local-llm},{Key=Project,Value=ai-codec-v3}]' \
    --region $REGION \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✅ Instance launched: $INSTANCE_ID"
echo ""

# Wait for instance to be running
echo "⏳ Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION
echo "✅ Instance is running!"
echo ""

# Get instance IP
INSTANCE_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PrivateIpAddress' \
    --output text)

echo "✅ Instance IP: $INSTANCE_IP"
echo ""

# Wait for instance to be ready for SSM
echo "⏳ Waiting for instance to be ready for SSM commands..."
sleep 30
aws ssm wait command-executed --command-id $(aws ssm send-command \
    --instance-ids $INSTANCE_ID \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["echo ready"]' \
    --region $REGION \
    --query 'Command.CommandId' \
    --output text) \
    --instance-id $INSTANCE_ID \
    --region $REGION 2>/dev/null || true
echo "✅ Instance ready!"
echo ""

# Upload setup script
echo "📤 Uploading setup script..."
aws s3 cp ../deploy/setup_local_llm.sh s3://ai-codec-v3-artifacts-580473065386/code/setup_local_llm.sh
SETUP_URL=$(aws s3 presign s3://ai-codec-v3-artifacts-580473065386/code/setup_local_llm.sh --expires-in 3600)

# Run setup
echo "🔧 Running setup on instance (this will take 10-15 minutes)..."
echo "   Downloading and installing:"
echo "   - Python 3.11"
echo "   - vLLM"
echo "   - Llama 3.1 8B model (~16GB download)"
echo ""

COMMAND_ID=$(aws ssm send-command \
    --instance-ids $INSTANCE_ID \
    --document-name "AWS-RunShellScript" \
    --parameters "{\"commands\":[\"cd /home/ubuntu\",\"curl -L -o setup.sh '$SETUP_URL'\",\"chmod +x setup.sh\",\"./setup.sh 2>&1 | tee setup.log\"]}" \
    --region $REGION \
    --query 'Command.CommandId' \
    --output text)

echo "Command ID: $COMMAND_ID"
echo ""
echo "⏳ Setup running... (check progress with: tail -f setup.log on instance)"
echo ""

# Save instance info
cat > local_llm_instance.json << EOF
{
  "instance_id": "$INSTANCE_ID",
  "instance_ip": "$INSTANCE_IP",
  "instance_type": "$INSTANCE_TYPE",
  "region": "$REGION",
  "llm_url": "http://$INSTANCE_IP:8000/v1/chat/completions",
  "cost_per_hour": 0.526,
  "setup_command_id": "$COMMAND_ID"
}
EOF

echo "✅ Instance info saved to: local_llm_instance.json"
echo ""
echo "============================================================"
echo "📋 NEXT STEPS:"
echo "============================================================"
echo ""
echo "1. Wait for setup to complete (~15 minutes)"
echo "   Check: aws ssm get-command-invocation --command-id $COMMAND_ID --instance-id $INSTANCE_ID --region $REGION"
echo ""
echo "2. Start vLLM server:"
echo "   aws ssm send-command --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript --parameters 'commands=[\"nohup /home/ubuntu/start_vllm.sh > vllm.log 2>&1 &\"]' --region $REGION"
echo ""
echo "3. Test LLM:"
echo "   curl http://$INSTANCE_IP:8000/v1/models"
echo ""
echo "4. Deploy orchestrator:"
echo "   Update worker orchestrator to point to: http://$INSTANCE_IP:8000"
echo ""
echo "💰 Cost: \$0.526/hour (~\$12.62/day)"
echo "⚡ Speed: 5-10x faster than Claude, NO rate limits!"
echo ""
echo "============================================================"

