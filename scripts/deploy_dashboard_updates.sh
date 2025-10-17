#!/bin/bash
# Deploy dashboard updates to S3/CloudFront

set -e

echo "================================================================"
echo "Deploying Dashboard Updates"
echo "================================================================"
echo ""
echo "Changes:"
echo "  ✓ Public dashboard now matches admin dashboard layout"
echo "  ✓ Added Media column with video and decoder downloads"
echo "  ✓ Added Runtime, Code, Version, GitHub columns"
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ ERROR: AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

echo "✅ AWS credentials verified"
echo ""

# Get S3 bucket and CloudFront distribution
REGION="us-east-1"
BUCKET_NAME="your-dashboard-bucket"  # Replace with actual bucket
CLOUDFRONT_ID="YOUR_DISTRIBUTION_ID"  # Replace with actual distribution ID

echo "📦 Uploading dashboard files..."
cd dashboard

# Upload updated JavaScript files
echo "  Uploading app.js (public dashboard)..."
aws s3 cp app.js s3://${BUCKET_NAME}/app.js --region ${REGION}

echo "  Uploading admin.js (admin dashboard)..."
aws s3 cp admin.js s3://${BUCKET_NAME}/admin.js --region ${REGION}

echo "  ✅ Files uploaded to S3"
echo ""

# Invalidate CloudFront cache
echo "🔄 Invalidating CloudFront cache..."
aws cloudfront create-invalidation \
    --distribution-id ${CLOUDFRONT_ID} \
    --paths "/app.js" "/admin.js" \
    --region ${REGION} \
    --output json > /dev/null 2>&1 || echo "  ⚠️  CloudFront invalidation skipped (check distribution ID)"

echo ""
echo "================================================================"
echo "Deployment Complete!"
echo "================================================================"
echo ""
echo "What changed:"
echo ""
echo "Public Dashboard (index.html):"
echo "  • Added Runtime column (elapsed vs estimated)"
echo "  • Added Code column (✨ LLM badge)"
echo "  • Added Version column (v1, v2, etc.)"
echo "  • Added Git column (commit hash or local)"
echo "  • Added Media column (🎬 Video + 💾 Decoder downloads)"
echo ""
echo "Admin Dashboard (admin.html):"
echo "  • Same as public dashboard PLUS:"
echo "  • Keeps Rerun button (admin-only)"
echo "  • Keeps Bug Analysis column"
echo "  • Keeps Human Intervention column"
echo ""
echo "Media Downloads:"
echo "  • Video: Reconstructed video output from experiments"
echo "  • Decoder: Python decoder code to decompress data"
echo "  • Only shown for successful experiments"
echo ""
echo "Next steps:"
echo "  1. Visit your dashboard: https://<your-domain>/index.html"
echo "  2. Verify new columns are visible"
echo "  3. Test video and decoder download links"
echo "  4. Check admin dashboard has rerun button"
echo ""
echo "Documentation: DASHBOARD_PARITY_FIX.md"
echo ""


