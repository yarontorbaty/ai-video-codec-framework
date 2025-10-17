#!/bin/bash
# Setup GitHub credentials in AWS Secrets Manager
# Usage: ./setup_github_credentials.sh

set -e

echo "=================================================="
echo "GitHub Credentials Setup for LLM Commits"
echo "=================================================="
echo ""

# Get GitHub credentials from user
echo "Enter your GitHub username:"
read GITHUB_USERNAME

echo ""
echo "Enter your GitHub personal access token (PAT):"
echo "Create one at: https://github.com/settings/tokens"
echo "Required scopes: repo (full control)"
read -s GITHUB_TOKEN

echo ""
echo "Enter your GitHub email (for commits):"
read GITHUB_EMAIL

echo ""
echo "Enter repository URL (e.g., https://github.com/username/AiV1.git):"
read GITHUB_REPO

echo ""
echo "Storing credentials in AWS Secrets Manager..."

# Create secret in AWS Secrets Manager
aws secretsmanager create-secret \
    --name ai-video-codec/github-credentials \
    --description "GitHub credentials for LLM autonomous commits" \
    --secret-string "{\"username\":\"$GITHUB_USERNAME\",\"token\":\"$GITHUB_TOKEN\",\"email\":\"$GITHUB_EMAIL\",\"repo\":\"$GITHUB_REPO\"}" \
    --region us-east-1 2>/dev/null || \

aws secretsmanager update-secret \
    --secret-id ai-video-codec/github-credentials \
    --secret-string "{\"username\":\"$GITHUB_USERNAME\",\"token\":\"$GITHUB_TOKEN\",\"email\":\"$GITHUB_EMAIL\",\"repo\":\"$GITHUB_REPO\"}" \
    --region us-east-1

echo ""
echo "✅ GitHub credentials stored in Secrets Manager"

# Grant orchestrator instance permission to read the secret
echo ""
echo "Granting orchestrator permissions..."

ORCHESTRATOR_ROLE=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=ai-video-codec-orchestrator" "Name=instance-state-name,Values=running" \
    --query 'Reservations[0].Instances[0].IamInstanceProfile.Arn' \
    --output text | awk -F'/' '{print $NF}')

if [ ! -z "$ORCHESTRATOR_ROLE" ]; then
    ROLE_NAME=$(aws iam get-instance-profile \
        --instance-profile-name "$ORCHESTRATOR_ROLE" \
        --query 'InstanceProfile.Roles[0].RoleName' \
        --output text)
    
    echo "Orchestrator role: $ROLE_NAME"
    
    # Create policy for GitHub secret access
    aws iam put-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-name GitHubSecretsAccess \
        --policy-document '{
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "secretsmanager:GetSecretValue"
                    ],
                    "Resource": "arn:aws:secretsmanager:us-east-1:*:secret:ai-video-codec/github-credentials-*"
                }
            ]
        }'
    
    echo "✅ Permissions granted"
else
    echo "⚠️  Could not find orchestrator role. You may need to grant permissions manually."
fi

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "The LLM can now:"
echo "  ✅ Commit code changes to GitHub"
echo "  ✅ Push improvements autonomously"
echo "  ✅ Track evolution via git history"
echo ""
echo "Credentials stored in:"
echo "  aws secretsmanager get-secret-value --secret-id ai-video-codec/github-credentials --region us-east-1"
echo ""

