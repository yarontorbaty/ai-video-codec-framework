# Activate Cost Allocation Tags - Quick Guide

## Why This is Needed

AWS requires you to manually activate cost allocation tags before they appear in Cost Explorer. This is a one-time setup that takes 24 hours to take effect.

## Steps to Activate

### Option 1: AWS Console (Easiest)

1. **Open AWS Billing Console**:
   - Go to: https://console.aws.amazon.com/billing/home

2. **Navigate to Cost Allocation Tags**:
   - Click on "Cost Allocation Tags" in the left sidebar
   - Or go directly to: https://console.aws.amazon.com/billing/home#/tags

3. **Find the CostCategory Tag**:
   - Look for `CostCategory` in the list of available tags
   - It should show as "Inactive" status
   - It may take a few hours for newly created tags to appear in this list

4. **Activate the Tag**:
   - Check the box next to `CostCategory`
   - Click "Activate" button at the top

5. **Wait for Data**:
   - Cost data will start populating within 24 hours
   - Check your dashboard after 24 hours to see real cost breakdowns

### Option 2: AWS CLI

```bash
# Check if CostCategory tag is available
aws ce list-cost-allocation-tags --status Inactive

# Activate the CostCategory tag
aws ce update-cost-allocation-tags-status \
  --cost-allocation-tags-status \
  TagKey=CostCategory,Status=Active

# Verify activation
aws ce list-cost-allocation-tags --status Active
```

## What to Expect

### Before Activation (Current State)
- Dashboard shows **total monthly cost** ✅ (working now)
- Cost breakdown by category shows **$0.00** for individual categories
- This is because AWS hasn't started tracking costs by the tag yet

### After Activation (24 hours later)
- Dashboard shows **total monthly cost** ✅
- Cost breakdown shows **real costs** for each category:
  - Training: Actual cost from g5.4xlarge GPU instances
  - Inference: Actual cost from g4dn.xlarge instances
  - Storage: Actual cost from S3 buckets and EFS
  - Orchestrator: Actual cost from c6i.xlarge instance

### Timeline
```
Now                  +1 hour            +24 hours          +48 hours
 |                      |                    |                  |
 ├─ Tags activated      ├─ Tags visible     ├─ Cost data       ├─ Full cost
 │  in console          │  in Billing       │  starts          │  history
 │                      │  Dashboard         │  appearing       │  available
 └─ Resources          └─ No cost data     └─ Partial data    └─ Complete data
    are tagged            yet                  available          populated
```

## Verify Tags are Applied

### Check EC2 Instances
```bash
# Check orchestrator tags
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=ai-video-codec-orchestrator" \
  --query 'Reservations[*].Instances[*].Tags' \
  --output table

# Check training worker tags
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=ai-video-codec-training-worker" \
  --query 'Reservations[*].Instances[*].Tags' \
  --output table

# Check inference worker tags
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=ai-video-codec-inference-worker" \
  --query 'Reservations[*].Instances[*].Tags' \
  --output table
```

You should see `CostCategory` tag with values:
- `Orchestrator` for orchestrator instance
- `Training` for training workers
- `Inference` for inference workers

### Check S3 Buckets
```bash
# List all project buckets with tags
aws s3api list-buckets \
  --query 'Buckets[?starts_with(Name, `ai-video-codec`)].Name' \
  --output table

# Check tags for a specific bucket
aws s3api get-bucket-tagging \
  --bucket ai-video-codec-artifacts-580473065386
```

You should see `CostCategory: Storage` tag.

## Troubleshooting

### Tag doesn't appear in Cost Allocation Tags list
- **Wait**: New tags can take 2-4 hours to appear
- **Check resources**: Ensure resources are actually tagged
- **Verify deployment**: Confirm CloudFormation stacks updated successfully

### Cost data still shows $0 after 24 hours
- **Check tag activation**: Verify tag status is "Active"
- **Check date range**: Costs only populate from activation date forward
- **Check resources are running**: Stopped resources don't incur costs
- **Check tag spelling**: Must be exact match: `CostCategory` (case-sensitive)

### Dashboard shows error
- **Check Lambda logs**:
  ```bash
  aws logs tail /aws/lambda/ai-video-codec-production-dashboard-renderer --follow
  ```
- **Check IAM permissions**: Lambda needs `ce:GetCostAndUsage` permission
- **Check Cost Explorer API**: Ensure it's enabled in your account

## Cost Allocation Tag Best Practices

### Current Tags Applied
```yaml
Tags:
  - Key: Name
    Value: ai-video-codec-[resource-name]
  - Key: Environment
    Value: production
  - Key: Project
    Value: ai-video-codec
  - Key: CostCategory        # ← This is the cost allocation tag
    Value: [Orchestrator|Training|Inference|Storage]
  - Key: CostCenter
    Value: AI-Video-Codec
```

### Tag Values by Resource Type

| Resource Type | CostCategory Value |
|--------------|-------------------|
| Orchestrator EC2 | Orchestrator |
| Training Workers (g5.4xlarge) | Training |
| Inference Workers (g4dn.xlarge) | Inference |
| S3 Buckets | Storage |
| EFS File System | Storage |
| DynamoDB Tables | Orchestrator |
| SQS Queues | Orchestrator |

## Expected Cost Breakdown

Based on typical AI/ML workloads, you might see:

```
Total Monthly Cost: $679.26
├─ Training:      $475.48  (70%)  ← GPU instances for training
├─ Orchestrator:  $ 67.93  (10%)  ← Management overhead
├─ Inference:     $ 81.51  (12%)  ← Inference workers
└─ Storage:       $ 54.34  ( 8%)  ← S3, EFS, EBS
```

*Note: Actual percentages will vary based on your workload patterns*

## Additional Resources

- [AWS Cost Allocation Tags Documentation](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/cost-alloc-tags.html)
- [AWS Cost Explorer API Reference](https://docs.aws.amazon.com/aws-cost-management/latest/APIReference/API_GetCostAndUsage.html)
- [AWS Tagging Best Practices](https://docs.aws.amazon.com/whitepapers/latest/tagging-best-practices/tagging-best-practices.html)

## Quick Status Check

Run this command to see your current setup:

```bash
#!/bin/bash
echo "=== Cost Allocation Tag Status ==="
aws ce list-cost-allocation-tags --status Active --query 'CostAllocationTags[?TagKey==`CostCategory`]'
echo ""
echo "=== Tagged EC2 Instances ==="
aws ec2 describe-instances \
  --filters "Name=tag:CostCategory,Values=*" \
  --query 'Reservations[*].Instances[*].[Tags[?Key==`Name`].Value|[0],Tags[?Key==`CostCategory`].Value|[0],State.Name]' \
  --output table
echo ""
echo "=== Dashboard URL ==="
echo "https://aiv1codec.com"
```

---

**Action Required**: Please activate the `CostCategory` tag in AWS Billing Console!
**Time Required**: 5 minutes to activate, 24 hours for data to populate
**Benefit**: Real cost breakdown by resource type in your dashboard

