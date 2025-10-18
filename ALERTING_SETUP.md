# Neural Codec HTTP System - Alerting Setup

**Date:** 2025-10-17 13:16:00 UTC

## 🚨 Alerting Components

### ✅ SNS Topic
- **Name:** neural-codec-alerts
- **ARN:** arn:aws:sns:us-east-1:580473065386:neural-codec-alerts
- **Purpose:** Centralized alert notifications

### ✅ CloudWatch Alarms (with SNS notifications)
- `neural-codec-worker-cpu-high` - Worker CPU > 80%
- `neural-codec-orchestrator-cpu-high` - Orchestrator CPU > 80%
- `neural-codec-worker-status-check` - Worker health check failures
- `neural-codec-orchestrator-status-check` - Orchestrator health check failures

## 📧 Email Notifications

### Current Configuration:
- **Topic:** neural-codec-alerts
- **Protocol:** email
- **Endpoint:** admin@your-domain.com

### ⚠️ Important:
1. **Confirm Subscription:** Check your email and click the confirmation link
2. **Update Email:** Replace 'admin@your-domain.com' with your actual email address
3. **Add More Subscribers:** You can add additional email addresses in the AWS console

## 🔔 Alert Types

### Critical Alerts (Immediate notification):
- Service health check failures
- High CPU utilization (> 80%)
- Service unreachable

### Recovery Notifications:
- Alarms clearing (services back to normal)
- Health checks passing again

## 📱 Managing Alerts

### AWS Console:
- **SNS Topics:** https://us-east-1.console.aws.amazon.com/sns/v3/home?region=us-east-1#/topics
- **CloudWatch Alarms:** https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#alarmsV2:

### Add More Subscribers:
```bash
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:580473065386:neural-codec-alerts \
  --protocol email \
  --notification-endpoint your-email@domain.com
```

### Test Alerts:
```bash
aws sns publish \
  --topic-arn arn:aws:sns:us-east-1:580473065386:neural-codec-alerts \
  --message "Test alert from Neural Codec system" \
  --subject "Test Alert"
```

## 🎯 Next Steps

1. **Confirm email subscription** (check your email)
2. **Update email address** to your actual email
3. **Test alert system** using the commands above
4. **Monitor for alerts** during system operation

---
*Alerting setup completed successfully!*
