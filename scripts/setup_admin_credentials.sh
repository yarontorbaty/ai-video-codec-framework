#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AiV1 - Admin Credentials Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get username
echo -e "${YELLOW}Enter admin username:${NC}"
read -r USERNAME

# Get password
echo -e "${YELLOW}Enter admin password:${NC}"
read -s PASSWORD
echo ""

# Get email for 2FA
echo -e "${YELLOW}Enter admin email (for 2FA codes):${NC}"
read -r EMAIL

# Validate email format
if [[ ! "$EMAIL" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
    echo -e "${RED}✗ Invalid email format${NC}"
    exit 1
fi

# Confirm
echo ""
echo -e "${BLUE}Please confirm:${NC}"
echo -e "Username: ${GREEN}${USERNAME}${NC}"
echo -e "Email: ${GREEN}${EMAIL}${NC}"
echo ""
echo -e "${YELLOW}Is this correct? (y/n)${NC}"
read -r CONFIRM

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo -e "${RED}Setup cancelled${NC}"
    exit 0
fi

# Create JSON secret
SECRET_JSON=$(cat <<EOF
{
  "username": "$USERNAME",
  "password": "$PASSWORD",
  "email": "$EMAIL",
  "2fa_enabled": true
}
EOF
)

echo ""
echo -e "${BLUE}Storing credentials in AWS Secrets Manager...${NC}"

# Store in Secrets Manager
aws secretsmanager put-secret-value \
    --secret-id ai-video-codec/admin-credentials \
    --secret-string "$SECRET_JSON" \
    --region us-east-1 > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Credentials stored successfully${NC}"
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Setup Complete!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "You can now log in to the admin interface at:"
    echo -e "${GREEN}https://aiv1codec.com/admin.html${NC}"
    echo ""
    echo -e "Username: ${GREEN}${USERNAME}${NC}"
    echo -e "2FA will be sent to: ${GREEN}${EMAIL}${NC}"
    echo ""
else
    echo -e "${RED}✗ Failed to store credentials${NC}"
    echo -e "Make sure you have AWS CLI configured with proper credentials"
    exit 1
fi

# Check if SES email is verified
echo -e "${YELLOW}Checking SES email verification...${NC}"
VERIFIED=$(aws ses get-identity-verification-attributes \
    --identities "$EMAIL" \
    --region us-east-1 \
    --query "VerificationAttributes.\"$EMAIL\".VerificationStatus" \
    --output text 2>/dev/null || echo "NotFound")

if [ "$VERIFIED" != "Success" ]; then
    echo ""
    echo -e "${YELLOW}⚠ Email not verified in AWS SES${NC}"
    echo ""
    echo -e "To enable 2FA email sending, you need to verify your email:"
    echo ""
    echo -e "1. Run this command to start verification:"
    echo -e "${BLUE}aws ses verify-email-identity --email-address \"$EMAIL\" --region us-east-1${NC}"
    echo ""
    echo -e "2. Check your email and click the verification link"
    echo ""
    echo -e "3. After verification, 2FA emails will be sent automatically"
    echo ""
    echo -e "${YELLOW}Note: If you're in SES sandbox, you can only send to verified addresses.${NC}"
    echo -e "${YELLOW}Request production access here: https://console.aws.amazon.com/ses/${NC}"
else
    echo -e "${GREEN}✓ Email is verified in SES - 2FA ready!${NC}"
fi

echo ""
echo -e "${BLUE}Security Tips:${NC}"
echo -e "• Use a strong, unique password"
echo -e "• Keep your email secure"
echo -e "• 2FA codes expire after 10 minutes"
echo -e "• Sessions expire after 24 hours"
echo ""

