#!/usr/bin/env python3
"""
Setup Alerting for Neural Codec HTTP System
Creates SNS topics and configures CloudWatch alarms to send notifications.
"""

import boto3
import json

def create_sns_topic():
    """Create SNS topic for neural codec alerts."""
    sns = boto3.client('sns', region_name='us-east-1')
    
    try:
        response = sns.create_topic(
            Name='neural-codec-alerts',
            Attributes={
                'DisplayName': 'Neural Codec System Alerts'
            }
        )
        
        topic_arn = response['TopicArn']
        print(f"✅ Created SNS topic: {topic_arn}")
        
        # Create email subscription (you'll need to confirm this)
        try:
            email_subscription = sns.subscribe(
                TopicArn=topic_arn,
                Protocol='email',
                Endpoint='admin@your-domain.com'  # Replace with your email
            )
            print(f"📧 Email subscription created: {email_subscription['SubscriptionArn']}")
            print("   ⚠️  Check your email to confirm the subscription")
        except Exception as e:
            print(f"⚠️  Could not create email subscription: {e}")
            print("   You can add email subscriptions later in the AWS console")
        
        return topic_arn
        
    except Exception as e:
        print(f"❌ Failed to create SNS topic: {e}")
        return None

def update_cloudwatch_alarms(topic_arn):
    """Update CloudWatch alarms to send notifications to SNS topic."""
    cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
    
    alarms = [
        'neural-codec-worker-cpu-high',
        'neural-codec-orchestrator-cpu-high',
        'neural-codec-worker-status-check',
        'neural-codec-orchestrator-status-check'
    ]
    
    success_count = 0
    for alarm_name in alarms:
        try:
            # Get current alarm configuration
            response = cloudwatch.describe_alarms(AlarmNames=[alarm_name])
            if not response['MetricAlarms']:
                print(f"⚠️  Alarm {alarm_name} not found")
                continue
            
            alarm = response['MetricAlarms'][0]
            
            # Update alarm with SNS topic
            cloudwatch.put_metric_alarm(
                AlarmName=alarm_name,
                AlarmDescription=alarm['AlarmDescription'],
                MetricName=alarm['MetricName'],
                Namespace=alarm['Namespace'],
                Statistic=alarm['Statistic'],
                Dimensions=alarm['Dimensions'],
                Period=alarm['Period'],
                EvaluationPeriods=alarm['EvaluationPeriods'],
                Threshold=alarm['Threshold'],
                ComparisonOperator=alarm['ComparisonOperator'],
                AlarmActions=[topic_arn],
                OKActions=[topic_arn]  # Also notify when alarm clears
            )
            
            print(f"✅ Updated alarm {alarm_name} with SNS notifications")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Failed to update alarm {alarm_name}: {e}")
    
    print(f"📊 Updated {success_count}/{len(alarms)} alarms with SNS notifications")
    return success_count == len(alarms)

def create_alert_summary(topic_arn):
    """Create alerting setup summary."""
    from datetime import datetime
    summary = f"""# Neural Codec HTTP System - Alerting Setup

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

## 🚨 Alerting Components

### ✅ SNS Topic
- **Name:** neural-codec-alerts
- **ARN:** {topic_arn}
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
aws sns subscribe \\
  --topic-arn {topic_arn} \\
  --protocol email \\
  --notification-endpoint your-email@domain.com
```

### Test Alerts:
```bash
aws sns publish \\
  --topic-arn {topic_arn} \\
  --message "Test alert from Neural Codec system" \\
  --subject "Test Alert"
```

## 🎯 Next Steps

1. **Confirm email subscription** (check your email)
2. **Update email address** to your actual email
3. **Test alert system** using the commands above
4. **Monitor for alerts** during system operation

---
*Alerting setup completed successfully!*
"""
    
    with open('ALERTING_SETUP.md', 'w') as f:
        f.write(summary)
    
    print("✅ Alerting setup summary created: ALERTING_SETUP.md")

def main():
    """Main alerting setup function."""
    from datetime import datetime
    
    print("🚨 Setting up Alerting for Neural Codec HTTP System")
    print("=" * 60)
    
    # 1. Create SNS topic
    print("\n1️⃣ Creating SNS topic...")
    topic_arn = create_sns_topic()
    
    if not topic_arn:
        print("❌ Failed to create SNS topic - cannot proceed with alarm updates")
        return
    
    # 2. Update CloudWatch alarms
    print("\n2️⃣ Updating CloudWatch alarms with SNS notifications...")
    if update_cloudwatch_alarms(topic_arn):
        print("✅ All alarms updated successfully")
    else:
        print("⚠️  Some alarms failed to update")
    
    # 3. Create alerting summary
    print("\n3️⃣ Creating alerting summary...")
    create_alert_summary(topic_arn)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Alerting setup complete!")
    print(f"📧 SNS Topic ARN: {topic_arn}")
    print("⚠️  Remember to confirm your email subscription!")
    print("📖 See ALERTING_SETUP.md for detailed information")

if __name__ == '__main__':
    main()
