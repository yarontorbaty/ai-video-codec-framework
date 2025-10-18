# v2.0 Pipeline Test Results

**Test Date:** October 17, 2025
**Test Type:** End-to-end pipeline connectivity test

---

## Test Steps Executed

1. ✅ **Purged SQS queue** - Cleared stuck messages
2. ✅ **Marked old experiment as failed** - Cleaned up DynamoDB  
3. ✅ **Sent test message #1** - Message ID: d7a582ec-28fa-4bdd-9f2a-38ac9d0cc619
4. ✅ **Sent test message #2** (after purge complete) - Message ID: 83b612c3-7eac-4a22-a0cf-4ae001b5c846
5. 🔄 **Restarted GPU worker** - To refresh state

---

## Observations

### SQS Queue Behavior
- Messages sent successfully
- Queue shows 2 messages "in-flight"  
- Worker logs show "No jobs available"

### GPU Worker
- Process running (PID 5989, then restarted)
- Polling every 20 seconds consistently
- Using correct queue URL
- IAM role: `ai-video-codec-production-compute-WorkerInstanceProfile-GSKDcgRmR9dX`

### Issue Identified
**The worker cannot see messages that SQS reports as "in-flight"**

Possible causes:
1. **IAM Permissions** - Worker role might not have SQS:ReceiveMessage permission
2. **Queue Configuration** - Visibility timeout or other settings
3. **Code Issue** - Worker's SQS receive logic might have a bug
4. **Multiple Consumers** - Another process might be receiving messages

---

## Next Debugging Steps

### 1. Check IAM Permissions
```bash
aws iam get-role-policy \
  --role-name $(aws iam list-instance-profiles | grep -A 5 "WorkerInstanceProfile" | grep "RoleName" | cut -d'"' -f4) \
  --policy-name SQSPolicy
```

### 2. Check Worker Code
Review the `poll_queue()` method in `neural_codec_gpu_worker.py` to ensure:
- Correct SQS receive_message call
- Proper error handling
- Message visibility timeout handling

### 3. Test SQS Manually from Worker Instance
```bash
aws ssm send-command \
  --instance-ids i-0b614aa221757060e \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["cd /home/ubuntu/ai-video-codec-framework && python3 -c \"import boto3; sqs=boto3.client('sqs'); r=sqs.receive_message(QueueUrl='https://sqs.us-east-1.amazonaws.com/580473065386/ai-video-codec-training-queue',MaxNumberOfMessages=1); print(r)\""]'
```

---

## Status

**Infrastructure:** ✅ Fully deployed and running  
**Code:** ✅ All v2.0 files in place  
**SQS Messaging:** ⚠️ Messages not reaching worker  
**Root Cause:** 🔍 Under investigation (likely IAM or code issue)

---

## Recommendation

1. Check IAM permissions on the worker role
2. Review worker's SQS polling code for bugs
3. Consider adding debug logging to worker's receive_message call

The infrastructure is solid - we just need to fix the SQS consumption issue.

