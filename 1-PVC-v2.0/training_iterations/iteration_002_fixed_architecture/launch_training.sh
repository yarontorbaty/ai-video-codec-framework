#!/bin/bash
# Automated Launch Script for True Autoencoder Training
# Usage: ./launch_training.sh <instance-id>

set -e

INSTANCE_ID=$1
REGION="us-east-1"
S3_BUCKET="s3://ai-codec-v3-artifacts-580473065386/pvc"

if [ -z "$INSTANCE_ID" ]; then
    echo "Usage: $0 <instance-id>"
    echo ""
    echo "Example: $0 i-1234567890abcdef0"
    echo ""
    echo "To launch a new instance first:"
    echo "  aws ec2 run-instances \\"
    echo "    --image-id ami-0c02fb55b15a6caa6 \\"
    echo "    --instance-type g5.48xlarge \\"
    echo "    --iam-instance-profile Name=EC2-SSM-S3-Full-Access \\"
    echo "    --block-device-mappings '[{\"DeviceName\":\"/dev/xda\",\"Ebs\":{\"VolumeSize\":100,\"VolumeType\":\"gp3\"}}]' \\"
    echo "    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=autoencoder-training}]' \\"
    echo "    --region us-east-1"
    exit 1
fi

echo "=========================================="
echo "AUTOENCODER TRAINING - AUTOMATED SETUP"
echo "=========================================="
echo "Instance: $INSTANCE_ID"
echo "Region: $REGION"
echo ""

# Step 1: Wait for instance to be ready
echo "Step 1/5: Waiting for instance to be ready..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION
echo "✓ Instance running"

# Step 2: Wait for SSM to be available
echo ""
echo "Step 2/5: Waiting for SSM agent (~2 minutes)..."
for i in {1..60}; do
    STATUS=$(aws ssm describe-instance-information \
        --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
        --region $REGION \
        --query 'InstanceInformationList[0].PingStatus' \
        --output text 2>/dev/null || echo "NotReady")
    
    if [ "$STATUS" == "Online" ]; then
        echo "✓ SSM agent online"
        break
    fi
    
    echo -n "."
    sleep 2
done

if [ "$STATUS" != "Online" ]; then
    echo "✗ SSM agent not responding after 2 minutes"
    exit 1
fi

# Step 3: Setup environment
echo ""
echo "Step 3/5: Setting up environment..."
CMD_ID=$(aws ssm send-command \
    --instance-ids $INSTANCE_ID \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "echo === Creating directories ===",
        "mkdir -p /home/ec2-user/autoencoder_training/models",
        "mkdir -p /home/ec2-user/autoencoder_training/trained_models",
        "cd /home/ec2-user/autoencoder_training",
        "echo === Downloading training files ===",
        "aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/training_iterations/iteration_001/ . --recursive --region us-east-1",
        "echo === Downloading dataset (71GB - will take ~15-20 minutes) ===",
        "aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/datasets/anime_frames_960x540_50k.npy . --region us-east-1",
        "echo === Installing dependencies ===",
        "pip3 install torch torchvision flask --quiet || true",
        "pip3 install \"urllib3<2.0\" --quiet || true",
        "echo === Setup complete ==="
    ]' \
    --region $REGION \
    --output text --query 'Command.CommandId')

echo "Setup command ID: $CMD_ID"
echo "Waiting for setup to complete (this will take ~20 minutes for dataset download)..."

# Wait for setup command to complete
for i in {1..600}; do
    STATUS=$(aws ssm get-command-invocation \
        --command-id "$CMD_ID" \
        --instance-id $INSTANCE_ID \
        --region $REGION \
        --query 'Status' \
        --output text 2>/dev/null || echo "InProgress")
    
    if [ "$STATUS" == "Success" ]; then
        echo "✓ Setup complete"
        break
    elif [ "$STATUS" == "Failed" ] || [ "$STATUS" == "Cancelled" ]; then
        echo "✗ Setup failed with status: $STATUS"
        aws ssm get-command-invocation \
            --command-id "$CMD_ID" \
            --instance-id $INSTANCE_ID \
            --region $REGION \
            --query 'StandardOutputContent' \
            --output text
        exit 1
    fi
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Still running... ($((i*2))s elapsed)"
    fi
    sleep 2
done

# Step 4: Start training
echo ""
echo "Step 4/5: Starting training..."
CMD_ID=$(aws ssm send-command \
    --instance-ids $INSTANCE_ID \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "cd /home/ec2-user/autoencoder_training",
        "rm -f training.log",
        "nohup python3 -u train_autoencoder_multigpu.py --dataset anime_frames_960x540_50k.npy --output-dir ./trained_models --epochs 100 --batch-size 8 --no-perceptual > training.log 2>&1 &",
        "sleep 5",
        "echo Training started",
        "tail -20 training.log"
    ]' \
    --region $REGION \
    --output text --query 'Command.CommandId')

sleep 10
aws ssm get-command-invocation \
    --command-id "$CMD_ID" \
    --instance-id $INSTANCE_ID \
    --region $REGION \
    --query 'StandardOutputContent' \
    --output text

echo "✓ Training started"

# Step 5: Start dashboard
echo ""
echo "Step 5/5: Starting web dashboard..."
CMD_ID=$(aws ssm send-command \
    --instance-ids $INSTANCE_ID \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "cd /home/ec2-user/autoencoder_training",
        "pkill -f autoencoder_dashboard.py || true",
        "PYTHONUNBUFFERED=1 python3 -u autoencoder_dashboard.py > dashboard.log 2>&1 &",
        "sleep 3",
        "curl -s http://localhost:8080 >/dev/null && echo Dashboard started || echo Dashboard not responding"
    ]' \
    --region $REGION \
    --output text --query 'Command.CommandId')

sleep 5
aws ssm get-command-invocation \
    --command-id "$CMD_ID" \
    --instance-id $INSTANCE_ID \
    --region $REGION \
    --query 'StandardOutputContent' \
    --output text

# Get instance IP
INSTANCE_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "=========================================="
echo "✅ TRAINING LAUNCHED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Instance IP: $INSTANCE_IP"
echo "Dashboard: http://$INSTANCE_IP:8080"
echo ""
echo "Monitor training:"
echo "  aws ssm send-command --instance-ids $INSTANCE_ID \\"
echo "    --document-name \"AWS-RunShellScript\" \\"
echo "    --parameters 'commands=[\"tail -100 /home/ec2-user/autoencoder_training/training.log | grep Epoch\"]' \\"
echo "    --region $REGION"
echo ""
echo "Training will complete in ~19 hours"
echo "=========================================="

