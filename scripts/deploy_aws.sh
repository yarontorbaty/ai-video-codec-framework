#!/bin/bash
set -e

# AI Video Codec Framework - AWS Deployment Script
# This script deploys the infrastructure to AWS using CloudFormation

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="ai-video-codec"
ENVIRONMENT="production"
REGION="us-east-1"
STACK_NAME="${PROJECT_NAME}-${ENVIRONMENT}"

# Get AWS Account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Video Codec Framework - AWS Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}Error: AWS CLI not configured or credentials not set${NC}"
    echo "Please run: aws configure"
    exit 1
fi

echo -e "${GREEN}✓ AWS CLI configured${NC}"
echo "Account ID: ${ACCOUNT_ID}"
echo "Region: ${REGION}"
echo ""

# Check if config file exists
if [ ! -f "config/aws_config.yaml" ]; then
    echo -e "${YELLOW}Warning: config/aws_config.yaml not found${NC}"
    echo "Creating from template..."
    cp config/aws_config.yaml.template config/aws_config.yaml
    echo -e "${YELLOW}Please edit config/aws_config.yaml with your values before continuing${NC}"
    echo "Required values:"
    echo "  - aws.account_id: ${ACCOUNT_ID}"
    echo "  - infrastructure.orchestrator.key_pair: Your EC2 key pair name"
    echo ""
    read -p "Press Enter to continue after editing the config file..."
fi

# Validate CloudFormation templates
echo -e "${BLUE}Validating CloudFormation templates...${NC}"
aws cloudformation validate-template --template-body file://infrastructure/cloudformation/compute.yaml --region ${REGION}
aws cloudformation validate-template --template-body file://infrastructure/cloudformation/storage.yaml --region ${REGION}
echo -e "${GREEN}✓ Templates validated${NC}"
echo ""

# Deploy storage stack first
echo -e "${BLUE}Deploying storage infrastructure...${NC}"
aws cloudformation deploy \
    --template-file infrastructure/cloudformation/storage.yaml \
    --stack-name ${STACK_NAME}-storage \
    --parameter-overrides \
        ProjectName=${PROJECT_NAME} \
        Environment=${ENVIRONMENT} \
        AccountId=${ACCOUNT_ID} \
    --capabilities CAPABILITY_IAM \
    --region ${REGION} \
    --no-fail-on-empty-changeset

echo -e "${GREEN}✓ Storage stack deployed${NC}"
echo ""

# Deploy compute stack
echo -e "${BLUE}Deploying compute infrastructure...${NC}"
aws cloudformation deploy \
    --template-file infrastructure/cloudformation/compute.yaml \
    --stack-name ${STACK_NAME}-compute \
    --parameter-overrides \
        ProjectName=${PROJECT_NAME} \
        Environment=${ENVIRONMENT} \
        KeyPairName="YOUR_KEY_PAIR_NAME" \
        OrchestratorInstanceType="c6i.xlarge" \
        TrainingWorkerInstanceType="g5.4xlarge" \
        InferenceWorkerInstanceType="g4dn.xlarge" \
        MaxTrainingWorkers=4 \
        MaxInferenceWorkers=4 \
    --capabilities CAPABILITY_IAM \
    --region ${REGION} \
    --no-fail-on-empty-changeset

echo -e "${GREEN}✓ Compute stack deployed${NC}"
echo ""

# Get outputs
echo -e "${BLUE}Getting deployment outputs...${NC}"
ORCHESTRATOR_IP=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME}-compute \
    --query 'Stacks[0].Outputs[?OutputKey==`OrchestratorPublicIP`].OutputValue' \
    --output text \
    --region ${REGION})

TRAINING_QUEUE_URL=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME}-compute \
    --query 'Stacks[0].Outputs[?OutputKey==`TrainingQueueUrl`].OutputValue' \
    --output text \
    --region ${REGION})

ARTIFACTS_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME}-storage \
    --query 'Stacks[0].Outputs[?OutputKey==`ArtifactsBucketName`].OutputValue' \
    --output text \
    --region ${REGION})

echo -e "${GREEN}✓ Deployment complete!${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Deployment Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Orchestrator IP: ${ORCHESTRATOR_IP}"
echo "Training Queue: ${TRAINING_QUEUE_URL}"
echo "Artifacts Bucket: ${ARTIFACTS_BUCKET}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. SSH to orchestrator: ssh ec2-user@${ORCHESTRATOR_IP}"
echo "2. Check orchestrator status: sudo systemctl status ai-video-codec-orchestrator"
echo "3. View logs: sudo journalctl -u ai-video-codec-orchestrator -f"
echo "4. Upload test videos to S3 bucket: ${ARTIFACTS_BUCKET}"
echo ""
echo -e "${BLUE}Monitoring:${NC}"
echo "- CloudWatch Logs: /aws/ec2/ai-video-codec-orchestrator"
echo "- Cost tracking: AWS Cost Explorer"
echo "- SQS queues: AWS SQS Console"
echo ""
echo -e "${GREEN}Deployment completed successfully!${NC}"
