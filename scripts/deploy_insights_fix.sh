#!/bin/bash
# Deploy the dashboard insights context fix to AWS Lambda

set -e

echo "================================================================"
echo "Deploying Dashboard Insights Context Fix"
echo "================================================================"
echo ""
echo "This script updates the blog dashboard Lambda function to properly"
echo "label LLM insights as 'pre-experiment analysis' rather than"
echo "'post-mortem analysis' to avoid confusion."
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ ERROR: AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

echo "✅ AWS credentials verified"
echo ""

# Get the Lambda function name
LAMBDA_FUNCTION="ai-video-codec-blog"
REGION="us-east-1"

# Check if function exists
if ! aws lambda get-function --function-name "$LAMBDA_FUNCTION" --region "$REGION" &> /dev/null; then
    echo "⚠️  WARNING: Lambda function '$LAMBDA_FUNCTION' not found in region '$REGION'"
    echo "   This fix has been saved to the source code and will be deployed"
    echo "   next time you update the dashboard stack."
    exit 0
fi

echo "📦 Creating deployment package..."
cd lambda

# Create a temporary directory for the package
TMP_DIR=$(mktemp -d)
echo "   Using temp directory: $TMP_DIR"

# Copy the Lambda function
cp index_ssr.py "$TMP_DIR/"

# Create ZIP file
cd "$TMP_DIR"
zip -q index_ssr.zip index_ssr.py
echo "   ✅ Package created: index_ssr.zip"

echo ""
echo "🚀 Deploying to Lambda..."
aws lambda update-function-code \
    --function-name "$LAMBDA_FUNCTION" \
    --zip-file "fileb://index_ssr.zip" \
    --region "$REGION" \
    --output json > /dev/null

if [ $? -eq 0 ]; then
    echo "   ✅ Lambda function updated successfully"
else
    echo "   ❌ Failed to update Lambda function"
    rm -rf "$TMP_DIR"
    exit 1
fi

# Cleanup
rm -rf "$TMP_DIR"
cd - > /dev/null

echo ""
echo "================================================================"
echo "Deployment Complete!"
echo "================================================================"
echo ""
echo "What changed:"
echo "  • 'Root Cause Analysis' → 'Pre-Experiment Analysis'"
echo "  • 'Key Insights' → 'Insights from Previous Experiments'"
echo "  • Added amber warning boxes with disclaimers"
echo "  • Made it clear these analyze PAST experiments, not current one"
echo ""
echo "Next steps:"
echo "  1. Visit your dashboard blog page to see the changes"
echo "  2. Read DASHBOARD_INSIGHTS_FIX.md for full details"
echo "  3. Consider implementing post-experiment analysis (see doc)"
echo ""
echo "Dashboard URL: https://<your-cloudfront-domain>/blog.html"
echo ""


