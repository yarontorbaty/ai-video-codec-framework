#!/bin/bash
# V3.0 Complete Deployment Script
# Deploys entire AI Video Codec Framework v3.0

set -e  # Exit on error

echo "================================================="
echo "AI Video Codec Framework v3.0 - Full Deployment"
echo "================================================="
echo ""

REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ANTHROPIC_SECRET="ai-video-codec-anthropic-key"

echo "📋 Configuration"
echo "---------------"
echo "Region: $REGION"
echo "Account: $ACCOUNT_ID"
echo ""

# Step 1: Deploy Database
echo "📋 Step 1/7: Deploying DynamoDB Table"
echo "--------------------------------------"
aws cloudformation deploy \
  --template-file v3/infrastructure/database.yaml \
  --stack-name ai-codec-v3-database \
  --region $REGION \
  --no-fail-on-empty-changeset
echo "✅ DynamoDB table deployed"
echo ""

# Step 2: Deploy Storage
echo "📋 Step 2/7: Deploying S3 Bucket"
echo "---------------------------------"
aws cloudformation deploy \
  --template-file v3/infrastructure/storage.yaml \
  --stack-name ai-codec-v3-storage \
  --region $REGION \
  --no-fail-on-empty-changeset
echo "✅ S3 bucket deployed"
echo ""

BUCKET_NAME="ai-codec-v3-artifacts-${ACCOUNT_ID}"
echo "Bucket: $BUCKET_NAME"
echo ""

# Step 3: Get default VPC and subnets
echo "📋 Step 3/7: Getting VPC Configuration"
echo "---------------------------------------"
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text --region $REGION)
SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[*].SubnetId' --output text --region $REGION)
SUBNET1=$(echo $SUBNET_IDS | cut -d' ' -f1)
SUBNET2=$(echo $SUBNET_IDS | cut -d' ' -f2)

echo "VPC: $VPC_ID"
echo "Orchestrator Subnet: $SUBNET1"
echo "Worker Subnet: $SUBNET2"
echo ""

# Step 4: Deploy EC2 Instances (simplified - manual for now)
echo "📋 Step 4/7: Launching EC2 Instances"
echo "-------------------------------------"

# Create security groups
ORCH_SG=$(aws ec2 create-security-group \
  --group-name ai-codec-v3-orchestrator \
  --description "Security group for orchestrator" \
  --vpc-id $VPC_ID \
  --region $REGION \
  --query 'GroupId' \
  --output text 2>/dev/null || \
  aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=ai-codec-v3-orchestrator" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --region $REGION)

WORKER_SG=$(aws ec2 create-security-group \
  --group-name ai-codec-v3-worker \
  --description "Security group for worker" \
  --vpc-id $VPC_ID \
  --region $REGION \
  --query 'GroupId' \
  --output text 2>/dev/null || \
  aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=ai-codec-v3-worker" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --region $REGION)

# Allow orchestrator to reach worker
aws ec2 authorize-security-group-ingress \
  --group-id $WORKER_SG \
  --protocol tcp \
  --port 8080 \
  --source-group $ORCH_SG \
  --region $REGION 2>/dev/null || true

echo "✅ Security groups ready"
echo "Orchestrator SG: $ORCH_SG"
echo "Worker SG: $WORKER_SG"
echo ""

# Step 5: Package and upload application code
echo "📋 Step 5/7: Packaging Application Code"
echo "----------------------------------------"

cd v3
tar -czf worker.tar.gz worker/
tar -czf orchestrator.tar.gz orchestrator/
cd ..

aws s3 cp v3/worker.tar.gz s3://$BUCKET_NAME/deploy/ --region $REGION
aws s3 cp v3/orchestrator.tar.gz s3://$BUCKET_NAME/deploy/ --region $REGION

echo "✅ Application code uploaded to S3"
echo ""

# Step 6: Create deployment summary
echo "📋 Step 6/7: Creating Deployment Summary"
echo "-----------------------------------------"

cat > v3_deployment_info.txt << EOF
AI Video Codec Framework v3.0 - Deployment Info
===============================================

Deployed: $(date)
Region: $REGION
Account: $ACCOUNT_ID

Resources:
- DynamoDB Table: ai-codec-v3-experiments
- S3 Bucket: $BUCKET_NAME
- VPC: $VPC_ID
- Orchestrator SG: $ORCH_SG
- Worker SG: $WORKER_SG

Next Steps:
1. Launch EC2 instances manually or via AWS Console
   - Orchestrator: t3.medium with SSM, use $ORCH_SG
   - Worker: g4dn.xlarge with SSM, use $WORKER_SG

2. Deploy code to instances:
   ./v3/deploy/deploy_to_instances.sh

3. Start services:
   - On orchestrator: cd /opt/orchestrator && python3 main.py
   - On worker: cd /opt/worker && python3 main.py

4. Monitor:
   - Check DynamoDB for experiments
   - Check S3 for videos
   - Check CloudWatch logs

Configuration:
- Anthropic Secret: $ANTHROPIC_SECRET
- Worker URL: http://[WORKER_PRIVATE_IP]:8080
EOF

cat v3_deployment_info.txt

echo ""
echo "✅ Deployment summary created"
echo ""

echo "================================================="
echo "✅ Infrastructure Deployment Complete!"
echo "================================================="
echo ""
echo "Summary saved to: v3_deployment_info.txt"
echo ""
echo "To complete deployment:"
echo "1. Launch EC2 instances (see v3_deployment_info.txt)"
echo "2. Run: ./v3/deploy/deploy_to_instances.sh"
echo ""

