# AWS Deployment Guide

This guide walks you through deploying the AI Video Codec Framework to AWS.

## 🚀 Quick Start

### Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
3. **EC2 Key Pair** created in your region
4. **Test Video** (4K60, 10 seconds) for experiments

### 1. Environment Setup

```bash
# Clone the repository
git clone git@github.com:yarontorbaty/ai-video-codec-framework.git
cd ai-video-codec-framework

# Set up development environment
./scripts/setup_environment.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Configure AWS

```bash
# Configure AWS CLI (if not already done)
aws configure

# Verify configuration
aws sts get-caller-identity
```

### 3. Create Configuration File

```bash
# Copy configuration template
cp config/aws_config.yaml.template config/aws_config.yaml

# Edit configuration
nano config/aws_config.yaml
```

**Required values to update:**
- `aws.account_id`: Your AWS account ID
- `infrastructure.orchestrator.key_pair`: Your EC2 key pair name
- `aws.region`: Your preferred region (default: us-east-1)

### 4. Deploy Infrastructure

```bash
# Deploy to AWS
./scripts/deploy_aws.sh
```

This will create:
- VPC with public/private subnets
- Security groups and IAM roles
- EC2 instances (orchestrator + workers)
- S3 buckets for storage
- DynamoDB tables for metadata
- SQS queues for messaging

### 5. Upload Test Data

```bash
# Upload test videos to S3
./scripts/upload_test_data.sh
```

### 6. Monitor Deployment

```bash
# Check status
./scripts/monitor_aws.sh

# Continuous monitoring
./scripts/monitor_aws.sh --watch
```

---

## 📋 Detailed Setup

### AWS Account Requirements

**IAM Permissions Needed:**
- EC2 (launch instances, create security groups)
- S3 (create buckets, upload/download objects)
- DynamoDB (create tables, read/write data)
- SQS (create queues, send/receive messages)
- SNS (create topics, publish messages)
- CloudFormation (create/update stacks)
- IAM (create roles and policies)
- VPC (create subnets, internet gateways)
- EFS (create file systems)
- KMS (create encryption keys)

**Cost Estimate:**
- Orchestrator (c6i.xlarge): ~$125/month
- Training workers (2× g5.4xlarge spot): ~$1,200/month
- Inference workers (1× g4dn.xlarge): ~$400/month
- Storage (S3 + EFS): ~$200/month
- **Total: ~$1,925/month** (well under $5,000 budget)

### Configuration Options

**Instance Types:**
- **Orchestrator**: c6i.large, c6i.xlarge, c6i.2xlarge
- **Training Workers**: g5.2xlarge, g5.4xlarge, g5.8xlarge, g5.12xlarge
- **Inference Workers**: g4dn.xlarge, g4dn.2xlarge, g4dn.4xlarge

**Scaling:**
- Training workers: 0-4 instances (auto-scaling)
- Inference workers: 0-4 instances (auto-scaling)
- Spot instances: 70% cost savings on training

**Storage:**
- S3 buckets: Artifacts, videos, models, reports
- EFS: Shared filesystem for experiments
- DynamoDB: Metadata and metrics

---

## 🏗️ Infrastructure Architecture

### Network Topology

```
Internet Gateway
       │
   ┌───▼───┐
   │  VPC  │
   │       │
   ├───────┤
   │Public │  ┌─────────────┐
   │Subnet │──│Orchestrator │
   │       │  │(c6i.xlarge)│
   └───────┘  └─────────────┘
       │
   ┌───▼───┐
   │Private│  ┌─────────────┐
   │Subnet │──│Training     │
   │       │  │Workers      │
   │       │  │(g5.4xlarge) │
   │       │  └─────────────┘
   │       │  ┌─────────────┐
   │       │──│Inference    │
   │       │  │Workers      │
   │       │  │(g4dn.xlarge)│
   └───────┘  └─────────────┘
```

### Storage Architecture

```
S3 Buckets:
├── artifacts-{account}     # Code, configs, logs
├── videos-{account}        # Source videos, references
├── models-{account}        # Trained models, checkpoints
└── reports-{account}       # Experiment reports, metrics

EFS File System:
└── /mnt/efs/experiments/   # Shared experiment data

DynamoDB Tables:
├── experiments            # Experiment metadata
├── metrics                # Performance metrics
└── cost-tracking         # Cost monitoring
```

### Security

**Network Security:**
- VPC with private subnets for workers
- Security groups with minimal access
- No public access to worker instances

**Data Security:**
- S3 buckets with encryption at rest
- EFS with KMS encryption
- IAM roles with least privilege
- No credentials stored in code

**Access Control:**
- SSH access only to orchestrator
- Workers accessible only from orchestrator
- S3 buckets private by default

---

## 🔧 Management Commands

### Deployment

```bash
# Deploy infrastructure
./scripts/deploy_aws.sh

# Update configuration
aws cloudformation deploy --template-file infrastructure/cloudformation/compute.yaml --stack-name ai-video-codec-production-compute

# Delete infrastructure (careful!)
aws cloudformation delete-stack --stack-name ai-video-codec-production-compute
aws cloudformation delete-stack --stack-name ai-video-codec-production-storage
```

### Monitoring

```bash
# Check orchestrator status
./scripts/monitor_aws.sh

# SSH to orchestrator
ssh ec2-user@$(aws cloudformation describe-stacks --stack-name ai-video-codec-production-compute --query 'Stacks[0].Outputs[?OutputKey==`OrchestratorPublicIP`].OutputValue' --output text)

# View orchestrator logs
ssh ec2-user@ORCHESTRATOR_IP 'sudo journalctl -u ai-video-codec-orchestrator -f'

# Check SQS queues
aws sqs list-queues

# Check DynamoDB tables
aws dynamodb list-tables
```

### Data Management

```bash
# Upload test data
./scripts/upload_test_data.sh

# Download results
aws s3 sync s3://ai-video-codec-reports-{account}/ ./reports/

# Clean up old data
aws s3 rm s3://ai-video-codec-artifacts-{account}/ --recursive --exclude "*.json"
```

---

## 🚨 Troubleshooting

### Common Issues

**1. Deployment Fails**
```bash
# Check CloudFormation events
aws cloudformation describe-stack-events --stack-name ai-video-codec-production-compute

# Check IAM permissions
aws sts get-caller-identity
aws iam list-attached-user-policies --user-name YOUR_USERNAME
```

**2. Orchestrator Not Starting**
```bash
# SSH to orchestrator
ssh ec2-user@ORCHESTRATOR_IP

# Check service status
sudo systemctl status ai-video-codec-orchestrator

# View logs
sudo journalctl -u ai-video-codec-orchestrator -f

# Check Python environment
python3 --version
pip3 list
```

**3. Workers Not Scaling**
```bash
# Check Auto Scaling Groups
aws autoscaling describe-auto-scaling-groups

# Check SQS queue
aws sqs get-queue-attributes --queue-url QUEUE_URL --attribute-names All

# Check CloudWatch metrics
aws cloudwatch get-metric-statistics --namespace AWS/AutoScaling --metric-name GroupDesiredCapacity
```

**4. High Costs**
```bash
# Check current costs
aws ce get-cost-and-usage --time-period Start=2025-10-01,End=2025-11-01 --granularity MONTHLY --metrics UnblendedCost

# Scale down workers
aws autoscaling update-auto-scaling-group --auto-scaling-group-name ai-video-codec-training-workers --desired-capacity 0

# Stop orchestrator
ssh ec2-user@ORCHESTRATOR_IP 'sudo systemctl stop ai-video-codec-orchestrator'
```

### Log Locations

**Orchestrator Logs:**
- System logs: `sudo journalctl -u ai-video-codec-orchestrator`
- Application logs: `/opt/ai-video-codec-framework/logs/`
- CloudWatch: `/aws/ec2/ai-video-codec-orchestrator`

**Worker Logs:**
- System logs: `sudo journalctl -u ai-video-codec-worker`
- Application logs: `/opt/ai-video-codec-framework/logs/`
- CloudWatch: `/aws/ec2/ai-video-codec-worker`

**S3 Logs:**
- Access logs: `s3://ai-video-codec-logs-{account}/`
- Experiment logs: `s3://ai-video-codec-artifacts-{account}/logs/`

---

## 📊 Monitoring & Alerts

### CloudWatch Metrics

**EC2 Metrics:**
- CPU utilization
- Memory utilization
- Network I/O
- Disk I/O

**SQS Metrics:**
- Queue depth
- Message age
- Throughput

**DynamoDB Metrics:**
- Read/write capacity
- Throttled requests
- Item count

**S3 Metrics:**
- Request count
- Data transfer
- Error rate

### Cost Alerts

**Budget Alerts:**
- 70% of budget: Scale down workers
- 85% of budget: Emergency stop
- 95% of budget: Shutdown all

**Cost Optimization:**
- Spot instances for training (70% savings)
- Auto-scaling based on queue depth
- S3 lifecycle policies
- EFS performance tuning

### Performance Monitoring

**Experiment Metrics:**
- Training time per experiment
- Model convergence rate
- Quality metrics (PSNR, SSIM)
- Compression ratios

**System Metrics:**
- Queue processing time
- Worker utilization
- Storage usage
- Network throughput

---

## 🔄 Maintenance

### Regular Tasks

**Daily:**
- Check orchestrator status
- Monitor experiment progress
- Review cost reports
- Check error logs

**Weekly:**
- Update AMI images
- Clean up old experiments
- Review performance metrics
- Update documentation

**Monthly:**
- Review and optimize costs
- Update security patches
- Backup important data
- Plan capacity changes

### Updates

**Framework Updates:**
```bash
# SSH to orchestrator
ssh ec2-user@ORCHESTRATOR_IP

# Pull latest code
cd /opt/ai-video-codec-framework
git pull origin main

# Restart services
sudo systemctl restart ai-video-codec-orchestrator
```

**Infrastructure Updates:**
```bash
# Update CloudFormation stack
aws cloudformation deploy --template-file infrastructure/cloudformation/compute.yaml --stack-name ai-video-codec-production-compute
```

---

## 📚 Additional Resources

### AWS Documentation
- [EC2 User Guide](https://docs.aws.amazon.com/ec2/)
- [S3 User Guide](https://docs.aws.amazon.com/s3/)
- [DynamoDB Developer Guide](https://docs.aws.amazon.com/dynamodb/)
- [CloudFormation User Guide](https://docs.aws.amazon.com/cloudformation/)

### Framework Documentation
- [README.md](README.md) - Project overview
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Technical details
- [TIMELINE_AND_MILESTONES.md](TIMELINE_AND_MILESTONES.md) - Project timeline

### Support
- GitHub Issues: [Create an issue](https://github.com/yarontorbaty/ai-video-codec-framework/issues)
- AWS Support: [AWS Support Center](https://console.aws.amazon.com/support/)
- Documentation: [Project Wiki](https://github.com/yarontorbaty/ai-video-codec-framework/wiki)

---

**Happy Deploying! 🚀**
