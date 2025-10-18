#!/bin/bash
set -e

REGION="us-east-1"
ORCH_SG="sg-0e573e2f685e36cb9"
WORKER_SG="sg-0885eababf6f844ba"
SUBNET1="subnet-439af579"
SUBNET2="subnet-6b25a31c"
AMI="ami-057a9f77fd28e08c5"
ROLE="ai-codec-v3-ec2-role"

echo "🚀 Launching Orchestrator..."
ORCH_ID=$(aws ec2 run-instances   --image-id $AMI   --instance-type t3.medium   --security-group-ids $ORCH_SG   --subnet-id $SUBNET1   --iam-instance-profile Name=$ROLE   --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ai-codec-v3-orchestrator}]'   --region $REGION   --query 'Instances[0].InstanceId'   --output text)

echo "✅ Orchestrator: $ORCH_ID"

echo "🚀 Launching Worker..."
WORKER_ID=$(aws ec2 run-instances   --image-id $AMI   --instance-type g4dn.xlarge   --security-group-ids $WORKER_SG   --subnet-id $SUBNET2   --iam-instance-profile Name=$ROLE   --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ai-codec-v3-worker}]'   --region $REGION   --query 'Instances[0].InstanceId'   --output text)

echo "✅ Worker: $WORKER_ID"

echo "⏳ Waiting for instances..."
aws ec2 wait instance-running --instance-ids $ORCH_ID $WORKER_ID --region $REGION
aws ec2 wait instance-status-ok --instance-ids $ORCH_ID $WORKER_ID --region $REGION

ORCH_IP=$(aws ec2 describe-instances --instance-ids $ORCH_ID --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text --region $REGION)
WORKER_IP=$(aws ec2 describe-instances --instance-ids $WORKER_ID --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text --region $REGION)

cat > v3_instances.txt << INST
ORCHESTRATOR_ID=$ORCH_ID
ORCHESTRATOR_IP=$ORCH_IP
WORKER_ID=$WORKER_ID
WORKER_IP=$WORKER_IP
INST

echo ""
echo "✅ Instances ready!"
echo "Orchestrator: $ORCH_ID ($ORCH_IP)"
echo "Worker: $WORKER_ID ($WORKER_IP)"
