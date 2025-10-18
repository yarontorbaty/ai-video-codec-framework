#!/bin/bash
set -e

echo "🚀 Launching EC2 Instances for v3.0"
echo "====================================="

REGION="us-east-1"
ORCH_SG="sg-0e573e2f685e36cb9"
WORKER_SG="sg-0885eababf6f844ba"
SUBNET1="subnet-439af579"
SUBNET2="subnet-6b25a31c"
AMI="ami-0c55b159cbfafe1f0"

# Get SSM-enabled IAM role (create if doesn't exist)
ROLE_NAME="ai-codec-v3-ec2-role"
aws iam get-role --role-name $ROLE_NAME 2>/dev/null || \
  aws iam create-role --role-name $ROLE_NAME \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam attach-role-policy --role-name $ROLE_NAME \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore 2>/dev/null || true

aws iam attach-role-policy --role-name $ROLE_NAME \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess 2>/dev/null || true

aws iam attach-role-policy --role-name $ROLE_NAME \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess 2>/dev/null || true

aws iam attach-role-policy --role-name $ROLE_NAME \
  --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite 2>/dev/null || true

# Create instance profile
aws iam create-instance-profile --instance-profile-name $ROLE_NAME 2>/dev/null || true
aws iam add-role-to-instance-profile --instance-profile-name $ROLE_NAME --role-name $ROLE_NAME 2>/dev/null || true

sleep 10  # Wait for role propagation

# Launch Orchestrator
echo "🚀 Launching Orchestrator..."
ORCH_ID=$(aws ec2 run-instances \
  --image-id $AMI \
  --instance-type t3.medium \
  --security-group-ids $ORCH_SG \
  --subnet-id $SUBNET1 \
  --iam-instance-profile Name=$ROLE_NAME \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ai-codec-v3-orchestrator}]' \
  --region $REGION \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "✅ Orchestrator launched: $ORCH_ID"

# Launch Worker
echo "🚀 Launching Worker..."
WORKER_ID=$(aws ec2 run-instances \
  --image-id $AMI \
  --instance-type g4dn.xlarge \
  --security-group-ids $WORKER_SG \
  --subnet-id $SUBNET2 \
  --iam-instance-profile Name=$ROLE_NAME \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ai-codec-v3-worker}]' \
  --region $REGION \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "✅ Worker launched: $WORKER_ID"

echo ""
echo "⏳ Waiting for instances to be running..."
aws ec2 wait instance-running --instance-ids $ORCH_ID $WORKER_ID --region $REGION

echo "✅ Instances are running"

# Get IPs
ORCH_IP=$(aws ec2 describe-instances --instance-ids $ORCH_ID --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text --region $REGION)
WORKER_IP=$(aws ec2 describe-instances --instance-ids $WORKER_ID --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text --region $REGION)

echo ""
echo "Instance Details:"
echo "  Orchestrator: $ORCH_ID ($ORCH_IP)"
echo "  Worker: $WORKER_ID ($WORKER_IP)"

# Save to file
cat > v3_instances.txt << INSTANCES
Orchestrator ID: $ORCH_ID
Orchestrator IP: $ORCH_IP
Worker ID: $WORKER_ID
Worker IP: $WORKER_IP
INSTANCES

echo ""
echo "✅ Instance info saved to v3_instances.txt"
