#!/bin/bash
set -e

# AI Video Codec Framework - Lambda Dashboard Deployment Script
# This script packages and deploys the Lambda function for SSR dashboard

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FUNCTION_NAME="ai-video-codec-dashboard-renderer"
REGION="us-east-1"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AiV1 - Lambda Dashboard Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}Error: AWS CLI not configured${NC}"
    exit 1
fi

echo -e "${GREEN}✓ AWS CLI configured${NC}"

# Check if Lambda source exists
if [ ! -f "lambda/index_ssr.py" ]; then
    echo -e "${RED}Error: Lambda source file not found (lambda/index_ssr.py)${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

echo -e "${GREEN}✓ Lambda source file found${NC}"

# Create deployment package
echo -e "${BLUE}Creating deployment package...${NC}"
cd lambda

# Clean up old package
rm -f lambda_dashboard.zip

# Create zip file
zip -q lambda_dashboard.zip index_ssr.py

echo -e "${GREEN}✓ Deployment package created ($(du -h lambda_dashboard.zip | cut -f1))${NC}"

# Get current Lambda configuration
echo -e "${BLUE}Checking Lambda function...${NC}"
LAMBDA_EXISTS=$(aws lambda get-function \
    --function-name ${FUNCTION_NAME} \
    --region ${REGION} 2>&1 || echo "NOT_FOUND")

if echo "$LAMBDA_EXISTS" | grep -q "Function not found"; then
    echo -e "${RED}Error: Lambda function ${FUNCTION_NAME} does not exist${NC}"
    echo -e "${YELLOW}Please create it first using CloudFormation or the AWS Console${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Lambda function exists${NC}"

# Deploy to Lambda
echo -e "${BLUE}Deploying to Lambda...${NC}"
aws lambda update-function-code \
    --function-name ${FUNCTION_NAME} \
    --zip-file fileb://lambda_dashboard.zip \
    --region ${REGION} \
    --output json > /tmp/lambda_update.json

# Extract info
NEW_CODE_SHA=$(cat /tmp/lambda_update.json | jq -r '.CodeSha256')
NEW_CODE_SIZE=$(cat /tmp/lambda_update.json | jq -r '.CodeSize')
LAST_MODIFIED=$(cat /tmp/lambda_update.json | jq -r '.LastModified')

echo -e "${GREEN}✓ Lambda function updated${NC}"
echo -e "  Code SHA256: ${NEW_CODE_SHA}"
echo -e "  Code Size: ${NEW_CODE_SIZE} bytes"
echo -e "  Last Modified: ${LAST_MODIFIED}"

# Wait for Lambda to be ready
echo -e "${BLUE}Waiting for Lambda to be ready...${NC}"
sleep 3

# Test invoke
echo -e "${BLUE}Testing Lambda function...${NC}"
TEST_EVENT='{
  "requestContext": {
    "http": {
      "method": "GET",
      "path": "/health"
    }
  },
  "headers": {
    "host": "aiv1codec.com"
  }
}'

aws lambda invoke \
    --function-name ${FUNCTION_NAME} \
    --payload "$TEST_EVENT" \
    --region ${REGION} \
    /tmp/lambda_test_response.json > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Lambda function responding${NC}"
else
    echo -e "${YELLOW}⚠ Lambda test failed, but code is deployed${NC}"
fi

# Clean up
rm -f lambda_dashboard.zip
cd ..

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Lambda deployment complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${BLUE}Function Name:${NC} ${FUNCTION_NAME}"
echo -e "${BLUE}Region:${NC} ${REGION}"
echo -e "${BLUE}Code SHA256:${NC} ${NEW_CODE_SHA}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. CloudFront cache may need 1-2 minutes to update"
echo "2. Test the dashboard at: https://aiv1codec.com/"
echo "3. Hard refresh (Ctrl+Shift+R) to bypass browser cache"
echo "4. Check CloudWatch Logs if issues persist"
echo ""
echo -e "${GREEN}Lambda is now updated! 🚀${NC}"

