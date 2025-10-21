#!/bin/bash
#
# Launch PVC Production Quick Test Training on AWS
#
# This script:
# 1. Launches a g4dn.xlarge GPU instance
# 2. Sets up the environment
# 3. Starts training
# 4. Monitors progress
#
# Expected cost: ~$10 (8-12 hours)
# Expected result: 27-28 dB PSNR validation
#

set -e

echo "========================================="
echo "PVC Production Quick Test - AWS Launch"
echo "========================================="
echo ""

# Configuration
INSTANCE_TYPE="g4dn.xlarge"
AMI_ID="ami-09e2639b59ee94f7c"  # Deep Learning AMI (Ubuntu 20.04)
KEY_NAME="ai-codec-key"
SECURITY_GROUP="sg-0a687ceae9f23fcd6"
IAM_PROFILE="SSM-InstanceProfile"
REGION="us-east-1"

echo "📋 Configuration:"
echo "   Instance Type: $INSTANCE_TYPE"
echo "   Region: $REGION"
echo "   Estimated Cost: ~$10"
echo "   Estimated Time: 8-12 hours"
echo ""

# Check if instance already running
echo "🔍 Checking for existing PVC training instances..."
EXISTING_INSTANCE=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=pvc-production-quick-test" \
            "Name=instance-state-name,Values=running,pending" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region $REGION 2>/dev/null)

if [ "$EXISTING_INSTANCE" != "None" ] && [ -n "$EXISTING_INSTANCE" ]; then
    echo "⚠️  Instance already running: $EXISTING_INSTANCE"
    echo ""
    read -p "Terminate and relaunch? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🛑 Terminating existing instance..."
        aws ec2 terminate-instances --instance-ids $EXISTING_INSTANCE --region $REGION
        echo "⏳ Waiting for termination..."
        aws ec2 wait instance-terminated --instance-ids $EXISTING_INSTANCE --region $REGION
        echo "✅ Instance terminated"
    else
        echo "❌ Aborted"
        exit 1
    fi
fi

# Launch instance
echo ""
echo "🚀 Launching GPU instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type $INSTANCE_TYPE \
  --iam-instance-profile Name=$IAM_PROFILE \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=pvc-production-quick-test},{Key=Project,Value=PVC-v2},{Key=Purpose,Value=QuickTest}]" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":100,\"VolumeType\":\"gp3\"}}]" \
  --query 'Instances[0].InstanceId' \
  --output text \
  --region $REGION)

echo "✅ Instance launched: $INSTANCE_ID"
echo ""

# Wait for instance to be running
echo "⏳ Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION
echo "✅ Instance running"
echo ""

# Wait for SSM to be available
echo "⏳ Waiting for SSM agent (this may take 2-3 minutes)..."
sleep 60

for i in {1..30}; do
    SSM_STATUS=$(aws ssm describe-instance-information \
      --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
      --query 'InstanceInformationList[0].PingStatus' \
      --output text \
      --region $REGION 2>/dev/null || echo "None")
    
    if [ "$SSM_STATUS" = "Online" ]; then
        echo "✅ SSM agent online"
        break
    fi
    
    echo "   Attempt $i/30: SSM status = $SSM_STATUS"
    sleep 10
done

if [ "$SSM_STATUS" != "Online" ]; then
    echo "❌ SSM agent did not come online"
    echo "   You may need to connect manually"
    exit 1
fi

echo ""
echo "📦 Setting up environment..."

# Create setup script
cat > /tmp/setup_pvc_training.sh << 'EOF'
#!/bin/bash
set -e

echo "Setting up PVC training environment..."

# Activate conda environment
source /opt/conda/bin/activate pytorch

# Install dependencies
pip install opencv-python-headless scikit-image tqdm

# Create working directory
mkdir -p /home/ec2-user/pvc_training
cd /home/ec2-user/pvc_training

echo "Environment setup complete"
EOF

# Upload and run setup
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters "commands=['bash -s < /tmp/setup_pvc_training.sh']" \
  --region $REGION \
  --output text > /tmp/setup_command_id.txt

echo "✅ Setup initiated"
echo ""

# Package PVC code
echo "📦 Packaging PVC code..."
cd "$(dirname "$0")/../pvc_v2"
tar czf /tmp/pvc_v2_code.tar.gz .
echo "✅ Code packaged"
echo ""

# Upload to S3
echo "📤 Uploading code to S3..."
aws s3 cp /tmp/pvc_v2_code.tar.gz s3://ai-codec-v3-artifacts-580473065386/pvc/training/pvc_v2_code.tar.gz
echo "✅ Code uploaded"
echo ""

# Download and extract on instance
echo "📥 Downloading code to instance..."
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["cd /home/ec2-user/pvc_training && aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/training/pvc_v2_code.tar.gz . && tar xzf pvc_v2_code.tar.gz"]' \
  --region $REGION \
  --output text > /tmp/download_command_id.txt

sleep 10
echo "✅ Code downloaded"
echo ""

# Start training
echo "🚀 Starting training..."
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["source /opt/conda/bin/activate pytorch && cd /home/ec2-user/pvc_training && nohup python3 training/train_production_quick.py --samples 10000 --epochs 100 --batch-size 8 --save-path /home/ec2-user/pvc_training/models > training.log 2>&1 &"]' \
  --region $REGION \
  --output text > /tmp/train_command_id.txt

echo "✅ Training started!"
echo ""
echo "========================================="
echo "Training Launched Successfully!"
echo "========================================="
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Region: $REGION"
echo ""
echo "📊 Monitor training:"
echo "   aws ssm start-session --target $INSTANCE_ID --region $REGION"
echo "   Then: tail -f /home/ec2-user/pvc_training/training.log"
echo ""
echo "📥 Download models when complete:"
echo "   aws s3 sync s3://ai-codec-v3-artifacts-580473065386/pvc/models/ ./models/"
echo ""
echo "💰 Cost estimate: ~$1.20/hour × 10 hours = $12"
echo ""
echo "⏰ Started: $(date)"
echo "⏰ Expected completion: $(date -v+12H 2>/dev/null || date -d '+12 hours')"
echo ""
echo "🛑 To stop training:"
echo "   aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""

