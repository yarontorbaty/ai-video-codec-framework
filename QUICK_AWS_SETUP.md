# 🚀 Quick AWS Setup

AWS CLI has been installed successfully! Now you need to configure it with your AWS credentials.

## Step 1: Get AWS Credentials

You need to get your AWS Access Key ID and Secret Access Key from your AWS account:

1. **Go to AWS Console**: https://console.aws.amazon.com/
2. **Navigate to IAM**: Search for "IAM" in the services
3. **Go to Users**: Click on your username
4. **Security Credentials**: Click "Security credentials" tab
5. **Create Access Key**: Click "Create access key"
6. **Choose Use Case**: Select "Command Line Interface (CLI)"
7. **Download Keys**: Save the Access Key ID and Secret Access Key

## Step 2: Configure AWS CLI

Run this command and enter your credentials:

```bash
aws configure
```

You'll be prompted for:
- **AWS Access Key ID**: `AKIA...` (from step 1)
- **AWS Secret Access Key**: `...` (from step 1)
- **Default region**: `us-east-1` (recommended)
- **Default output format**: `json`

## Step 3: Test Configuration

```bash
aws sts get-caller-identity
```

This should return your account information.

## Step 4: Continue Deployment

Once configured, run:

```bash
./scripts/setup_aws.sh
```

This will:
- Verify your AWS connection
- Create required resources (key pairs, security groups)
- Update the configuration file
- Prepare for deployment

## Alternative: Environment Variables

If you prefer not to use `aws configure`, you can set environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

## Troubleshooting

### Access Denied
If you get access denied errors, you need IAM permissions for:
- CloudFormation
- EC2
- S3
- IAM
- Lambda
- API Gateway

### Region Issues
Make sure you're using `us-east-1` as the region for this deployment.

---

**Once AWS CLI is configured, we can proceed with the deployment! 🚀**
