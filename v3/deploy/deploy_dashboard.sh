#!/bin/bash
# Deploy V3.0 Dashboard Lambda

set -e

echo "🚀 Deploying V3.0 Dashboard Lambda"
echo "===================================="

REGION="us-east-1"
FUNCTION_NAME="ai-codec-v3-dashboard"
BUCKET="ai-codec-v3-artifacts-580473065386"

# Create deployment package
echo "📦 Creating deployment package..."
cd v3/lambda
zip -q dashboard.zip dashboard.py
cd ../..

# Upload to S3
echo "📤 Uploading to S3..."
aws s3 cp v3/lambda/dashboard.zip s3://$BUCKET/lambda/ --region $REGION

# Check if function exists
if aws lambda get-function --function-name $FUNCTION_NAME --region $REGION 2>/dev/null; then
    echo "🔄 Updating existing function..."
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --s3-bucket $BUCKET \
        --s3-key lambda/dashboard.zip \
        --region $REGION
else
    echo "✨ Creating new function..."
    
    # Create IAM role if it doesn't exist
    ROLE_NAME="ai-codec-v3-lambda-role"
    ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text 2>/dev/null || echo "")
    
    if [ -z "$ROLE_ARN" ]; then
        echo "📝 Creating IAM role..."
        aws iam create-role \
            --role-name $ROLE_NAME \
            --assume-role-policy-document '{
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }' \
            --region $REGION
        
        # Attach policies
        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole \
            --region $REGION
        
        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBReadOnlyAccess \
            --region $REGION
        
        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
            --region $REGION
        
        sleep 10  # Wait for role to propagate
        
        ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text)
    fi
    
    # Create Lambda function
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --runtime python3.9 \
        --role $ROLE_ARN \
        --handler dashboard.lambda_handler \
        --code S3Bucket=$BUCKET,S3Key=lambda/dashboard.zip \
        --timeout 30 \
        --memory-size 512 \
        --environment "Variables={DYNAMODB_TABLE=ai-codec-v3-experiments,S3_BUCKET=$BUCKET}" \
        --region $REGION
fi

# Update environment variables
echo "⚙️  Updating environment variables..."
aws lambda update-function-configuration \
    --function-name $FUNCTION_NAME \
    --environment "Variables={DYNAMODB_TABLE=ai-codec-v3-experiments,S3_BUCKET=$BUCKET}" \
    --region $REGION

# Create or update function URL
echo "🌐 Creating function URL..."
FUNCTION_URL=$(aws lambda create-function-url-config \
    --function-name $FUNCTION_NAME \
    --auth-type NONE \
    --region $REGION \
    --query 'FunctionUrl' \
    --output text 2>/dev/null || echo "")

if [ -z "$FUNCTION_URL" ]; then
    FUNCTION_URL=$(aws lambda get-function-url-config \
        --function-name $FUNCTION_NAME \
        --region $REGION \
        --query 'FunctionUrl' \
        --output text)
fi

# Add public invoke permission
aws lambda add-permission \
    --function-name $FUNCTION_NAME \
    --statement-id FunctionURLAllowPublicAccess \
    --action lambda:InvokeFunctionUrl \
    --principal "*" \
    --function-url-auth-type NONE \
    --region $REGION 2>/dev/null || echo "Permission already exists"

echo ""
echo "✅ Dashboard deployed successfully!"
echo ""
echo "🌐 Dashboard URL: $FUNCTION_URL"
echo ""
echo "Test it: curl $FUNCTION_URL"
echo ""

# Save URL to file
echo "$FUNCTION_URL" > v3_dashboard_url.txt
echo "URL saved to: v3_dashboard_url.txt"

