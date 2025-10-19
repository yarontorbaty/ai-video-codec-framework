#!/bin/bash
# Deploy Fast Experiment System for 10,000+ experiments/hour

set -e

PROJECT_NAME="ai-codec-v3-fast"
REGION="us-east-1"

echo "=========================================="
echo "Deploying Fast Experiment System"
echo "=========================================="

# 1. Create DynamoDB table for fast experiments
echo "1. Creating DynamoDB table..."
aws dynamodb create-table \
  --table-name ai-codec-v3-fast-experiments \
  --attribute-definitions \
    AttributeName=experiment_id,AttributeType=S \
    AttributeName=timestamp,AttributeType=N \
  --key-schema \
    AttributeName=experiment_id,KeyType=HASH \
    AttributeName=timestamp,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region ${REGION} 2>/dev/null || echo "Table already exists"

echo "✅ DynamoDB table ready"

# 2. Launch fast worker instance (c5.2xlarge for CPU speed)
echo "2. Launching fast worker instance..."

WORKER_USER_DATA=$(cat <<'EOF'
#!/bin/bash
# Fast worker setup
set -e

# Update system
yum update -y
yum install -y python3 python3-pip git

# Install Python packages
pip3 install boto3 numpy opencv-python anthropic requests

# Create worker directory
mkdir -p /home/ec2-user/fast-worker
cd /home/ec2-user/fast-worker

# Download worker code from GitHub or S3
# For now, we'll wait for manual upload

echo "Fast worker instance ready - upload code and start service"
EOF
)

WORKER_INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-0453ec754f44f9a4a \
  --instance-type c5.2xlarge \
  --key-name gpu-ec2 \
  --security-group-ids sg-0e573e2f685e36cb9 \
  --subnet-id subnet-2f7b1b4a \
  --iam-instance-profile Name=ai-codec-v3-ec2-role \
  --user-data "$WORKER_USER_DATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${PROJECT_NAME}-worker},{Key=Project,Value=${PROJECT_NAME}}]" \
  --region ${REGION} \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Fast Worker Instance: $WORKER_INSTANCE_ID"

# Wait for instance to be running
echo "Waiting for worker instance to start..."
aws ec2 wait instance-running --instance-ids $WORKER_INSTANCE_ID --region ${REGION}

# Get worker IP
WORKER_IP=$(aws ec2 describe-instances \
  --instance-ids $WORKER_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' \
  --output text \
  --region ${REGION})

echo "Worker IP: $WORKER_IP"

# 3. Launch fast orchestrator instance (t3.medium is enough)
echo "3. Launching fast orchestrator instance..."

ORCH_USER_DATA=$(cat <<EOF2
#!/bin/bash
# Fast orchestrator setup
set -e

# Update system
yum update -y
yum install -y python3 python3-pip git

# Install Python packages
pip3 install boto3 anthropic requests

# Create orchestrator directory
mkdir -p /home/ec2-user/fast-orchestrator
cd /home/ec2-user/fast-orchestrator

echo "Fast orchestrator instance ready - upload code and start service"
EOF2
)

ORCH_INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-0453ec754f44f9a4a \
  --instance-type t3.medium \
  --key-name gpu-ec2 \
  --security-group-ids sg-0e573e2f685e36cb9 \
  --subnet-id subnet-2f7b1b4a \
  --iam-instance-profile Name=ai-codec-v3-ec2-role \
  --user-data "$ORCH_USER_DATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${PROJECT_NAME}-orchestrator},{Key=Project,Value=${PROJECT_NAME}}]" \
  --region ${REGION} \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Fast Orchestrator Instance: $ORCH_INSTANCE_ID"

# Wait for orchestrator
echo "Waiting for orchestrator instance to start..."
aws ec2 wait instance-running --instance-ids $ORCH_INSTANCE_ID --region ${REGION}

ORCH_IP=$(aws ec2 describe-instances \
  --instance-ids $ORCH_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' \
  --output text \
  --region ${REGION})

echo "Orchestrator IP: $ORCH_IP"

# Save instance info
cat > fast_instances.txt <<EOF3
Orchestrator: $ORCH_INSTANCE_ID
Worker: $WORKER_INSTANCE_ID
WORKER_IP=$WORKER_IP
ORCH_IP=$ORCH_IP
EOF3

echo ""
echo "=========================================="
echo "✅ Fast System Deployed Successfully!"
echo "=========================================="
echo ""
echo "Orchestrator: $ORCH_INSTANCE_ID ($ORCH_IP)"
echo "Worker: $WORKER_INSTANCE_ID ($WORKER_IP)"
echo ""
echo "Next steps:"
echo "1. Upload fast worker code to worker instance"
echo "2. Upload fast orchestrator code to orchestrator instance"
echo "3. Start worker service"
echo "4. Start orchestrator with: python3 fast_orchestrator.py http://$WORKER_IP:8080"
echo ""
echo "Instance info saved to: fast_instances.txt"
echo ""

