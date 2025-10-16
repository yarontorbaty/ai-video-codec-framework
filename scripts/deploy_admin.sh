#!/bin/bash
# Deploy Admin Chat Interface

set -e

PROJECT_NAME="ai-video-codec"
REGION="us-east-1"

echo "=========================================="
echo "Deploying Admin Chat Interface"
echo "=========================================="

# Deploy CloudFormation stack
echo "1. Deploying CloudFormation stack..."
aws cloudformation deploy \
  --template-file infrastructure/cloudformation/admin-interface.yaml \
  --stack-name ${PROJECT_NAME}-admin \
  --parameter-overrides ProjectName=${PROJECT_NAME} \
  --capabilities CAPABILITY_IAM \
  --region ${REGION}

echo "✅ CloudFormation stack deployed"

# Get Lambda function name
FUNCTION_NAME="${PROJECT_NAME}-admin-api"

# Package and deploy Lambda code
echo "2. Deploying Lambda function code..."
cd lambda
zip -q admin_api.zip admin_api.py
aws lambda update-function-code \
  --function-name ${FUNCTION_NAME} \
  --zip-file fileb://admin_api.zip \
  --region ${REGION}
rm admin_api.zip
cd ..

echo "✅ Lambda function updated"

# Upload admin HTML/JS to S3
echo "3. Uploading admin interface to S3..."
BUCKET_NAME="${PROJECT_NAME}-dashboard-580473065386"
aws s3 cp dashboard/admin.html s3://${BUCKET_NAME}/admin.html
aws s3 cp dashboard/admin.js s3://${BUCKET_NAME}/admin.js

echo "✅ Admin interface uploaded"

# Get API endpoint
echo "4. Getting API endpoint..."
API_ENDPOINT=$(aws cloudformation describe-stacks \
  --stack-name ${PROJECT_NAME}-admin \
  --query 'Stacks[0].Outputs[?OutputKey==`AdminAPIEndpoint`].OutputValue' \
  --output text \
  --region ${REGION})

echo ""
echo "=========================================="
echo "✅ Admin Interface Deployed Successfully!"
echo "=========================================="
echo ""
echo "API Endpoint: ${API_ENDPOINT}"
echo "Dashboard URL: https://aiv1codec.com/admin.html"
echo ""
echo "Next steps:"
echo "1. Set ANTHROPIC_API_KEY in Lambda environment"
echo "2. (Optional) Set ADMIN_PASSWORD_HASH for authentication"
echo "3. Invalidate CloudFront cache"
echo ""

