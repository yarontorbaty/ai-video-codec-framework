#!/bin/bash
set -e

echo "========================================="
echo "Experiment Cleanup Deployment"
echo "========================================="
echo

REGION="us-east-1"
STACK_NAME="ai-video-codec-experiment-cleanup"
LAMBDA_FUNCTION="ai-video-codec-experiment-cleanup"

# Step 1: Deploy CloudFormation stack
echo "📦 Deploying CloudFormation stack..."
aws cloudformation deploy \
    --template-file infrastructure/cloudformation/experiment-cleanup.yaml \
    --stack-name "$STACK_NAME" \
    --capabilities CAPABILITY_NAMED_IAM \
    --region "$REGION"

if [ $? -eq 0 ]; then
    echo "✅ CloudFormation stack deployed"
else
    echo "❌ CloudFormation deployment failed"
    exit 1
fi

# Step 2: Package and upload Lambda code
echo
echo "📦 Packaging Lambda function..."
cd lambda
zip -q /tmp/cleanup_lambda.zip experiment_cleanup.py

echo "📤 Updating Lambda function code..."
aws lambda update-function-code \
    --function-name "$LAMBDA_FUNCTION" \
    --zip-file fileb:///tmp/cleanup_lambda.zip \
    --region "$REGION" > /dev/null

if [ $? -eq 0 ]; then
    echo "✅ Lambda function updated"
else
    echo "❌ Lambda update failed"
    exit 1
fi

cd ..

# Step 3: Test the function
echo
echo "🧪 Testing cleanup function..."
aws lambda invoke \
    --function-name "$LAMBDA_FUNCTION" \
    --region "$REGION" \
    --log-type Tail \
    /tmp/cleanup_test_output.json > /tmp/cleanup_test_invoke.json

# Extract logs
echo
echo "📋 Test output:"
cat /tmp/cleanup_test_output.json | python3 -m json.tool
echo

LOGS=$(cat /tmp/cleanup_test_invoke.json | python3 -c "import sys, json, base64; print(base64.b64decode(json.load(sys.stdin).get('LogResult', '')).decode('utf-8'))" 2>/dev/null || echo "No logs available")

if [ ! -z "$LOGS" ]; then
    echo "📋 Lambda logs:"
    echo "$LOGS"
fi

# Step 4: Verify schedule
echo
echo "⏰ Verifying CloudWatch Events schedule..."
RULE_INFO=$(aws events describe-rule --name experiment-cleanup-schedule --region "$REGION" 2>/dev/null)

if [ $? -eq 0 ]; then
    echo "✅ Schedule rule active:"
    echo "$RULE_INFO" | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"  Name: {d['Name']}\"); print(f\"  Schedule: {d['ScheduleExpression']}\"); print(f\"  State: {d['State']}\")"
else
    echo "⚠️  Could not verify schedule rule"
fi

echo
echo "========================================="
echo "✅ Deployment Complete"
echo "========================================="
echo
echo "The cleanup function will now run automatically every 5 minutes."
echo "It will:"
echo "  • Detect experiments stuck for > 5-15 minutes (depending on phase)"
echo "  • Mark them as 'timed_out'"
echo "  • Update blog with abandonment reason"
echo "  • Flag for human intervention"
echo
echo "Monitor logs:"
echo "  aws logs tail /aws/lambda/$LAMBDA_FUNCTION --follow --region $REGION"
echo

