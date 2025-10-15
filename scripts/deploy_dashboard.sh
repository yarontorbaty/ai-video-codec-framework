#!/bin/bash
set -e

# AI Video Codec Framework - Dashboard Deployment Script
# This script deploys the dashboard to AWS S3 + CloudFront

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

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Video Codec Framework - Dashboard Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}Error: AWS CLI not configured${NC}"
    exit 1
fi

echo -e "${GREEN}✓ AWS CLI configured${NC}"

# Check if dashboard files exist
if [ ! -f "dashboard/index.html" ]; then
    echo -e "${RED}Error: Dashboard files not found${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

echo -e "${GREEN}✓ Dashboard files found${NC}"

# Deploy CloudFormation stack for dashboard
echo -e "${BLUE}Deploying dashboard infrastructure...${NC}"
aws cloudformation deploy \
    --template-file infrastructure/cloudformation/dashboard.yaml \
    --stack-name ${STACK_NAME}-dashboard \
    --parameter-overrides \
        ProjectName=${PROJECT_NAME} \
        Environment=${ENVIRONMENT} \
    --capabilities CAPABILITY_IAM \
    --region ${REGION} \
    --no-fail-on-empty-changeset

echo -e "${GREEN}✓ Dashboard infrastructure deployed${NC}"

# Get S3 bucket name
BUCKET_NAME=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME}-dashboard \
    --query 'Stacks[0].Outputs[?OutputKey==`DashboardBucketName`].OutputValue' \
    --output text \
    --region ${REGION})

if [ "$BUCKET_NAME" = "None" ] || [ -z "$BUCKET_NAME" ]; then
    echo -e "${RED}Error: Could not get S3 bucket name${NC}"
    exit 1
fi

echo -e "${GREEN}✓ S3 bucket: ${BUCKET_NAME}${NC}"

# Upload dashboard files to S3
echo -e "${BLUE}Uploading dashboard files...${NC}"

# Upload HTML files
aws s3 cp dashboard/index.html s3://${BUCKET_NAME}/index.html \
    --content-type "text/html" \
    --cache-control "no-cache" \
    --region ${REGION}

# Upload CSS files
aws s3 cp dashboard/styles.css s3://${BUCKET_NAME}/styles.css \
    --content-type "text/css" \
    --cache-control "max-age=86400" \
    --region ${REGION}

# Upload JavaScript files
aws s3 cp dashboard/app.js s3://${BUCKET_NAME}/app.js \
    --content-type "application/javascript" \
    --cache-control "max-age=86400" \
    --region ${REGION}

echo -e "${GREEN}✓ Dashboard files uploaded${NC}"

# Get CloudFront distribution ID
DISTRIBUTION_ID=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME}-dashboard \
    --query 'Stacks[0].Outputs[?OutputKey==`CloudFrontDistributionId`].OutputValue' \
    --output text \
    --region ${REGION})

if [ "$DISTRIBUTION_ID" != "None" ] && [ -n "$DISTRIBUTION_ID" ]; then
    echo -e "${BLUE}Invalidating CloudFront cache...${NC}"
    aws cloudfront create-invalidation \
        --distribution-id ${DISTRIBUTION_ID} \
        --paths "/*" \
        --region ${REGION} > /dev/null
    
    echo -e "${GREEN}✓ CloudFront cache invalidated${NC}"
fi

# Get dashboard URL
DASHBOARD_URL=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME}-dashboard \
    --query 'Stacks[0].Outputs[?OutputKey==`DashboardURL`].OutputValue' \
    --output text \
    --region ${REGION})

API_URL=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME}-dashboard \
    --query 'Stacks[0].Outputs[?OutputKey==`DashboardAPIGatewayURL`].OutputValue' \
    --output text \
    --region ${REGION})

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Dashboard deployment complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${BLUE}Dashboard URL:${NC}"
echo -e "${GREEN}${DASHBOARD_URL}${NC}"
echo ""
echo -e "${BLUE}API Endpoints:${NC}"
echo -e "${GREEN}${API_URL}/metrics${NC}"
echo -e "${GREEN}${API_URL}/experiments${NC}"
echo -e "${GREEN}${API_URL}/costs${NC}"
echo ""
echo -e "${BLUE}Features:${NC}"
echo "✓ Real-time metrics display"
echo "✓ Infrastructure status monitoring"
echo "✓ Cost tracking and breakdown"
echo "✓ Experiment progress tracking"
echo "✓ Responsive design for mobile/desktop"
echo "✓ Auto-refresh every 30 seconds"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Open the dashboard URL in your browser"
echo "2. Monitor real-time progress and costs"
echo "3. Share the URL with stakeholders"
echo "4. Bookmark for easy access"
echo ""
echo -e "${GREEN}Dashboard is now live! 🚀${NC}"
