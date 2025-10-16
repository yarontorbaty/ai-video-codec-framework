#!/bin/bash
set -e

# AI Video Codec Framework - AWS Setup Script
# This script helps set up AWS CLI and configuration

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Video Codec Framework - AWS Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${YELLOW}AWS CLI not found. Installing...${NC}"
    
    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            echo -e "${BLUE}Installing AWS CLI via Homebrew...${NC}"
            brew install awscli
        else
            echo -e "${YELLOW}Homebrew not found. Installing via pip...${NC}"
            pip3 install awscli
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo -e "${BLUE}Installing AWS CLI for Linux...${NC}"
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip awscliv2.zip
        sudo ./aws/install
        rm -rf awscliv2.zip aws/
    else
        echo -e "${RED}Unsupported OS. Please install AWS CLI manually.${NC}"
        echo "Visit: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
        exit 1
    fi
else
    echo -e "${GREEN}✓ AWS CLI already installed${NC}"
fi

# Check AWS CLI version
AWS_VERSION=$(aws --version 2>&1 | grep -oP '\d+\.\d+\.\d+' | head -1)
echo -e "${BLUE}AWS CLI version: ${AWS_VERSION}${NC}"

# Check if AWS is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${YELLOW}AWS CLI not configured. Please configure your credentials:${NC}"
    echo ""
    echo -e "${BLUE}Run: aws configure${NC}"
    echo ""
    echo "You'll need:"
    echo "- AWS Access Key ID"
    echo "- AWS Secret Access Key"
    echo "- Default region (us-east-1 recommended)"
    echo "- Default output format (json)"
    echo ""
    echo -e "${YELLOW}Press Enter when you've configured AWS CLI...${NC}"
    read -r
fi

# Test AWS connection
echo -e "${BLUE}Testing AWS connection...${NC}"
if aws sts get-caller-identity &> /dev/null; then
    echo -e "${GREEN}✓ AWS connection successful${NC}"
    
    # Get account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    echo -e "${BLUE}AWS Account ID: ${ACCOUNT_ID}${NC}"
    
    # Get region
    REGION=$(aws configure get region)
    echo -e "${BLUE}AWS Region: ${REGION}${NC}"
    
else
    echo -e "${RED}AWS connection failed. Please check your credentials.${NC}"
    exit 1
fi

# Check for required resources
echo -e "${BLUE}Checking for required AWS resources...${NC}"

# Check EC2 key pairs
KEY_PAIRS=$(aws ec2 describe-key-pairs --query 'KeyPairs[].KeyName' --output text 2>/dev/null || echo "")
if [ -z "$KEY_PAIRS" ]; then
    echo -e "${YELLOW}No EC2 key pairs found. Creating one...${NC}"
    aws ec2 create-key-pair --key-name ai-video-codec-key --query 'KeyMaterial' --output text > ~/.ssh/ai-video-codec-key.pem
    chmod 400 ~/.ssh/ai-video-codec-key.pem
    echo -e "${GREEN}✓ Created key pair: ai-video-codec-key${NC}"
    KEY_PAIR_NAME="ai-video-codec-key"
else
    echo -e "${GREEN}✓ Found existing key pairs: ${KEY_PAIRS}${NC}"
    KEY_PAIR_NAME=$(echo "$KEY_PAIRS" | head -n1)
fi

# Check security groups
SECURITY_GROUPS=$(aws ec2 describe-security-groups --query 'SecurityGroups[?GroupName==`ai-video-codec-sg`].GroupId' --output text 2>/dev/null || echo "")
if [ -z "$SECURITY_GROUPS" ]; then
    echo -e "${YELLOW}No security group found. Creating one...${NC}"
    SECURITY_GROUP_ID=$(aws ec2 create-security-group --group-name ai-video-codec-sg --description "Security group for AI Video Codec Framework" --query 'GroupId' --output text)
    
    # Add SSH access
    aws ec2 authorize-security-group-ingress --group-id "$SECURITY_GROUP_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0
    echo -e "${GREEN}✓ Added SSH access (port 22)${NC}"
    
    # Add HTTP access
    aws ec2 authorize-security-group-ingress --group-id "$SECURITY_GROUP_ID" --protocol tcp --port 80 --cidr 0.0.0.0/0
    echo -e "${GREEN}✓ Added HTTP access (port 80)${NC}"
    
    # Add HTTPS access
    aws ec2 authorize-security-group-ingress --group-id "$SECURITY_GROUP_ID" --protocol tcp --port 443 --cidr 0.0.0.0/0
    echo -e "${GREEN}✓ Added HTTPS access (port 443)${NC}"
    
    echo -e "${GREEN}✓ Created security group: ${SECURITY_GROUP_ID}${NC}"
else
    echo -e "${GREEN}✓ Found existing security group: ${SECURITY_GROUPS}${NC}"
    SECURITY_GROUP_ID="$SECURITY_GROUPS"
fi

# Update configuration file
echo -e "${BLUE}Updating configuration file...${NC}"
if [ -f "config/aws_config.yaml" ]; then
    # Update existing configuration
    python3 -c "
import yaml
import sys

# Load existing config
with open('config/aws_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update values
config['aws']['account_id'] = '$ACCOUNT_ID'
config['aws']['region'] = '$REGION'
config['infrastructure']['orchestrator']['key_pair'] = '$KEY_PAIR_NAME'
config['infrastructure']['orchestrator']['security_group'] = '$SECURITY_GROUP_ID'

# Save updated config
with open('config/aws_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print('Configuration updated successfully')
"
    echo -e "${GREEN}✓ Configuration updated${NC}"
else
    echo -e "${YELLOW}Configuration file not found. Please create it manually.${NC}"
    echo "Copy config/aws_config.yaml.template to config/aws_config.yaml and update the values."
fi

# Test required permissions
echo -e "${BLUE}Testing required AWS permissions...${NC}"

# Test CloudFormation
if aws cloudformation list-stacks --max-items 1 &> /dev/null; then
    echo -e "${GREEN}✓ CloudFormation access${NC}"
else
    echo -e "${RED}✗ CloudFormation access denied${NC}"
fi

# Test EC2
if aws ec2 describe-instances --max-items 1 &> /dev/null; then
    echo -e "${GREEN}✓ EC2 access${NC}"
else
    echo -e "${RED}✗ EC2 access denied${NC}"
fi

# Test S3
if aws s3 ls &> /dev/null; then
    echo -e "${GREEN}✓ S3 access${NC}"
else
    echo -e "${RED}✗ S3 access denied${NC}"
fi

# Test IAM
if aws iam get-user &> /dev/null; then
    echo -e "${GREEN}✓ IAM access${NC}"
else
    echo -e "${RED}✗ IAM access denied${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}AWS Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Deploy infrastructure: ./scripts/deploy_aws.sh"
echo "2. Deploy dashboard: ./scripts/deploy_dashboard.sh"
echo ""
echo -e "${BLUE}Configuration Summary:${NC}"
echo "- Account ID: ${ACCOUNT_ID}"
echo "- Region: ${REGION}"
echo "- Key Pair: ${KEY_PAIR_NAME}"
echo "- Security Group: ${SECURITY_GROUP_ID}"
echo ""
echo -e "${GREEN}Ready to deploy! 🚀${NC}"
