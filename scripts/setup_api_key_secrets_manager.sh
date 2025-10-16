#!/bin/bash
# One-time setup: Store Anthropic API key in AWS Secrets Manager
# This persists across all EC2 deployments

set -e

echo "=================================================="
echo "Setup Anthropic API Key in AWS Secrets Manager"
echo "=================================================="
echo ""
echo "This will store your API key securely in AWS."
echo "EC2 instances will retrieve it automatically."
echo ""

# Prompt for API key
read -p "Enter your Anthropic API key: " API_KEY

if [ -z "$API_KEY" ]; then
    echo "Error: API key cannot be empty"
    exit 1
fi

# Create or update secret
echo ""
echo "Storing API key in AWS Secrets Manager..."

aws secretsmanager create-secret \
    --name ai-video-codec/anthropic-api-key \
    --description "Anthropic API key for AI Video Codec LLM orchestrator" \
    --secret-string "{\"ANTHROPIC_API_KEY\":\"$API_KEY\"}" \
    --tags Key=Project,Value=ai-video-codec Key=Purpose,Value=LLM \
    2>/dev/null && echo "✅ Secret created successfully" || {
    
    # If secret exists, update it
    aws secretsmanager update-secret \
        --secret-id ai-video-codec/anthropic-api-key \
        --secret-string "{\"ANTHROPIC_API_KEY\":\"$API_KEY\"}" \
        && echo "✅ Secret updated successfully"
}

echo ""
echo "✅ Setup complete!"
echo ""
echo "Now updating IAM role to allow EC2 to read this secret..."

# Add permission to orchestrator role
ROLE_NAME="ai-video-codec-orchestrator-role"

aws iam put-role-policy \
    --role-name $ROLE_NAME \
    --policy-name SecretsManagerAccess \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "secretsmanager:GetSecretValue",
                    "secretsmanager:DescribeSecret"
                ],
                "Resource": "arn:aws:secretsmanager:us-east-1:*:secret:ai-video-codec/*"
            }
        ]
    }' && echo "✅ IAM permissions updated"

echo ""
echo "=================================================="
echo "✅ All done! Your API key is now stored securely."
echo "=================================================="
echo ""
echo "EC2 instances will automatically retrieve it on startup."
echo "You never need to set it again manually."

