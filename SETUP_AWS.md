# 🚀 AWS Setup Guide

Before deploying the AI Video Codec Framework, you need to set up AWS CLI and configure your credentials.

## Prerequisites

1. **AWS Account**: You need an active AWS account
2. **AWS CLI**: Install the AWS Command Line Interface
3. **Credentials**: Configure AWS access keys

## Step 1: Install AWS CLI

### macOS (using Homebrew)
```bash
brew install awscli
```

### macOS (using pip)
```bash
pip3 install awscli
```

### Linux/Windows
```bash
# Download and install from AWS
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

## Step 2: Configure AWS Credentials

### Option A: AWS Configure (Recommended)
```bash
aws configure
```

You'll be prompted for:
- **AWS Access Key ID**: Your access key
- **AWS Secret Access Key**: Your secret key  
- **Default region**: `us-east-1` (recommended)
- **Default output format**: `json`

### Option B: Environment Variables
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### Option C: AWS Profiles
```bash
aws configure --profile ai-video-codec
export AWS_PROFILE=ai-video-codec
```

## Step 3: Verify Setup

```bash
# Test AWS connection
aws sts get-caller-identity

# Should return your account ID and user info
```

## Step 4: Get Required Information

```bash
# Get your AWS account ID
aws sts get-caller-identity --query Account --output text

# List available regions
aws ec2 describe-regions --query 'Regions[].RegionName' --output table

# Check if you have EC2 key pairs
aws ec2 describe-key-pairs --query 'KeyPairs[].KeyName' --output table
```

## Step 5: Create Required Resources

### Create EC2 Key Pair (if needed)
```bash
# Create a new key pair
aws ec2 create-key-pair --key-name ai-video-codec-key --query 'KeyMaterial' --output text > ~/.ssh/ai-video-codec-key.pem
chmod 400 ~/.ssh/ai-video-codec-key.pem
```

### Create Security Group (if needed)
```bash
# Create security group
aws ec2 create-security-group --group-name ai-video-codec-sg --description "Security group for AI Video Codec Framework"

# Add SSH access
aws ec2 authorize-security-group-ingress --group-name ai-video-codec-sg --protocol tcp --port 22 --cidr 0.0.0.0/0

# Add HTTP access
aws ec2 authorize-security-group-ingress --group-name ai-video-codec-sg --protocol tcp --port 80 --cidr 0.0.0.0/0

# Add HTTPS access
aws ec2 authorize-security-group-ingress --group-name ai-video-codec-sg --protocol tcp --port 443 --cidr 0.0.0.0/0
```

## Step 6: Update Configuration

Once you have your AWS account ID and key pair name, update the configuration:

```bash
# Edit the configuration file
nano config/aws_config.yaml
```

Update these values:
```yaml
aws:
  account_id: "YOUR_ACCOUNT_ID"  # From aws sts get-caller-identity
  region: us-east-1

infrastructure:
  orchestrator:
    key_pair: "YOUR_KEY_PAIR_NAME"  # From aws ec2 describe-key-pairs
    security_group: "YOUR_SECURITY_GROUP_ID"  # From aws ec2 describe-security-groups
```

## Step 7: Test Deployment

```bash
# Test AWS connection
aws sts get-caller-identity

# Test EC2 access
aws ec2 describe-instances --max-items 1

# Test S3 access
aws s3 ls
```

## Troubleshooting

### AWS CLI Not Found
```bash
# Check if AWS CLI is installed
which aws

# If not found, install it
brew install awscli  # macOS
# or
pip3 install awscli
```

### Access Denied
```bash
# Check your permissions
aws iam get-user

# If you get access denied, you need IAM permissions for:
# - CloudFormation
# - EC2
# - S3
# - IAM
# - Lambda
# - API Gateway
```

### Region Issues
```bash
# Set default region
aws configure set region us-east-1

# Or use environment variable
export AWS_DEFAULT_REGION=us-east-1
```

## Required IAM Permissions

Your AWS user/role needs these permissions:
- `CloudFormation:*`
- `EC2:*`
- `S3:*`
- `IAM:*`
- `Lambda:*`
- `APIGateway:*`
- `CloudFront:*`
- `DynamoDB:*`
- `SQS:*`
- `EFS:*`
- `CloudWatch:*`

## Next Steps

Once AWS CLI is configured:

1. **Run the setup script**:
   ```bash
   ./scripts/setup_environment.sh
   ```

2. **Deploy the infrastructure**:
   ```bash
   ./scripts/deploy_aws.sh
   ```

3. **Deploy the dashboard**:
   ```bash
   ./scripts/deploy_dashboard.sh
   ```

## Support

If you encounter issues:
- Check AWS CLI version: `aws --version`
- Verify credentials: `aws sts get-caller-identity`
- Check permissions: `aws iam get-user`
- Review AWS documentation: https://docs.aws.amazon.com/cli/

---

**Ready to deploy once AWS CLI is configured! 🚀**
