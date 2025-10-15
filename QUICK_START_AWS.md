# 🚀 Quick Start - AWS Deployment

Get the AI Video Codec Framework running on AWS in 15 minutes!

## Prerequisites Checklist

- [ ] AWS account with billing enabled
- [ ] AWS CLI installed (`aws --version`)
- [ ] AWS CLI configured (`aws configure`)
- [ ] EC2 key pair created in your region
- [ ] Test video file (4K60, 10 seconds) - optional

## 🎯 5-Step Deployment

### Step 1: Clone & Setup
```bash
git clone git@github.com:yarontorbaty/ai-video-codec-framework.git
cd ai-video-codec-framework
./scripts/setup_environment.sh
source venv/bin/activate
```

### Step 2: Configure AWS
```bash
# Copy and edit config
cp config/aws_config.yaml.template config/aws_config.yaml
nano config/aws_config.yaml

# Required changes:
# - aws.account_id: YOUR_ACCOUNT_ID
# - infrastructure.orchestrator.key_pair: YOUR_KEY_PAIR_NAME
```

### Step 3: Deploy Infrastructure
```bash
./scripts/deploy_aws.sh
```
⏱️ **Takes ~10 minutes**

### Step 4: Upload Test Data
```bash
./scripts/upload_test_data.sh
```
📁 **Uploads your test video to S3**

### Step 5: Monitor & Verify
```bash
./scripts/monitor_aws.sh
```
📊 **Shows orchestrator status and queues**

---

## 🎉 You're Done!

Your framework is now running autonomously on AWS:

- **Orchestrator**: Managing experiments and workers
- **Training Workers**: Auto-scaling GPU instances
- **Storage**: S3 buckets for videos, models, and results
- **Monitoring**: Real-time cost and performance tracking

## 📊 What You Get

| Component | Instance Type | Purpose | Cost/Month |
|-----------|---------------|---------|------------|
| Orchestrator | c6i.xlarge | Master controller | ~$125 |
| Training Workers | g5.4xlarge (spot) | Model training | ~$1,200 |
| Inference Workers | g4dn.xlarge | Real-time processing | ~$400 |
| Storage | S3 + EFS | Data & models | ~$200 |
| **Total** | | | **~$1,925** |

**Budget Usage**: 38% of $5,000 monthly limit ✅

## 🔧 Next Steps

### Monitor Progress
```bash
# Real-time monitoring
./scripts/monitor_aws.sh --watch

# SSH to orchestrator
ssh ec2-user@$(aws cloudformation describe-stacks --stack-name ai-video-codec-production-compute --query 'Stacks[0].Outputs[?OutputKey==`OrchestratorPublicIP`].OutputValue' --output text)
```

### View Results
```bash
# Download experiment reports
aws s3 sync s3://ai-video-codec-reports-{account}/ ./reports/

# Check latest experiments
aws dynamodb scan --table-name ai-video-codec-experiments --limit 5
```

### Scale Resources
```bash
# Scale up training workers
aws autoscaling update-auto-scaling-group --auto-scaling-group-name ai-video-codec-training-workers --desired-capacity 4

# Scale down to save costs
aws autoscaling update-auto-scaling-group --auto-scaling-group-name ai-video-codec-training-workers --desired-capacity 0
```

## 🚨 Troubleshooting

### Common Issues

**Deployment fails?**
```bash
# Check CloudFormation events
aws cloudformation describe-stack-events --stack-name ai-video-codec-production-compute
```

**Orchestrator not starting?**
```bash
# SSH and check logs
ssh ec2-user@ORCHESTRATOR_IP
sudo journalctl -u ai-video-codec-orchestrator -f
```

**High costs?**
```bash
# Check current costs
aws ce get-cost-and-usage --time-period Start=2025-10-01,End=2025-11-01 --granularity MONTHLY --metrics UnblendedCost

# Scale down workers
aws autoscaling update-auto-scaling-group --auto-scaling-group-name ai-video-codec-training-workers --desired-capacity 0
```

## 📚 Documentation

- [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md) - Detailed deployment guide
- [README.md](README.md) - Project overview
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick commands
- [TIMELINE_AND_MILESTONES.md](TIMELINE_AND_MILESTONES.md) - Project timeline

## 🆘 Need Help?

- **GitHub Issues**: [Create an issue](https://github.com/yarontorbaty/ai-video-codec-framework/issues)
- **AWS Support**: [AWS Support Center](https://console.aws.amazon.com/support/)
- **Documentation**: [Project Wiki](https://github.com/yarontorbaty/ai-video-codec-framework/wiki)

---

**Ready to revolutionize video compression! 🎬🚀**
