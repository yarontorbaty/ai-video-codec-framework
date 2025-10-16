# Admin Chat Interface - Governing LLM Control

## Overview

The Admin Chat Interface allows you to interact directly with the governing LLM, provide suggestions, and control experiments in real-time.

## Access

**URL:** https://aiv1codec.com/admin.html

## Features

### 1. **Chat with Governing LLM**
- Natural language conversation with Claude Sonnet 4.5
- Ask questions about experiment status
- Provide suggestions for new approaches
- Get real-time insights and analysis

### 2. **Experiment Control**
- **Start New Experiment:** Trigger a new experiment immediately
- **Stop All Experiments:** Cancel running experiments
- **Pause Autonomous Mode:** Stop automatic experiment scheduling
- **Resume Autonomous:** Re-enable automatic experiments

### 3. **Real-Time Status**
- Total experiments run
- Currently running experiments
- Best bitrate achieved
- Success rate percentage
- System status indicator

### 4. **Quick Commands**
Predefined commands for common queries:
- "What is the current status?"
- "What should we try next?"
- "Analyze the last 3 experiments"
- "Try a hybrid approach with 70% procedural and 30% neural compression"
- "Focus on residual encoding"

## How to Use

### Basic Chat

1. Open https://aiv1codec.com/admin.html
2. Type your message in the chat input
3. Press Enter or click Send
4. The LLM will respond with insights and suggestions

### Example Conversations

**You:** "What's the current state of experiments?"
**LLM:** "We've run 14 experiments. The parameter storage approach is achieving 0.04 Mbps (99.6% compression), but it's not actually compressing real video - just storing generation parameters. The latest insight is that the codec architecture is inverted..."

**You:** "Try implementing residual encoding between procedural approximation and actual frames"
**LLM:** "Excellent suggestion! I'll generate a new experiment that: 1) Uses procedural generation as a base layer, 2) Encodes the residuals (differences), 3) Compresses residuals with neural networks..."

**You:** "Stop the current experiment and start a new one with different parameters"
**LLM:** "Stopping current experiments... Done. Starting new experiment with updated configuration..."

### Control Experiments

Use the control panel on the right to:
- Monitor system status (green = running, yellow = idle, red = stopped)
- Start/stop experiments with one click
- Pause/resume autonomous mode
- Configure LLM settings

### Settings

- **Auto-execute LLM suggestions:** Automatically run experiments suggested by LLM
- **Require approval for new code:** Manual approval before executing LLM-generated code
- **Verbose logging:** More detailed logs in the interface

## Architecture

### Frontend
- **Location:** `dashboard/admin.html`, `dashboard/admin.js`
- **Features:** Real-time chat UI, status monitoring, control buttons
- **Storage:** Chat history saved in localStorage

### Backend
- **Lambda:** `lambda/admin_api.py`
- **API Gateway:** `/admin/chat`, `/admin/status`, `/admin/command`
- **DynamoDB:** `ai-video-codec-control` table for state
- **SSM:** Controls EC2 orchestrator instance

### Security

1. **CORS:** Configured for https://aiv1codec.com only
2. **Authentication:** Optional password hash (set via `ADMIN_PASSWORD_HASH` env var)
3. **IAM Roles:** Least-privilege access for Lambda
4. **API Gateway:** Regional endpoint (not public edge)

## API Endpoints

### GET /admin/status
Returns current system status:
```json
{
  "total_experiments": 14,
  "running_now": 0,
  "best_bitrate": 0.04,
  "success_rate": 92.8,
  "autonomous_enabled": true
}
```

### POST /admin/chat
Chat with governing LLM:
```json
{
  "message": "What should we try next?",
  "history": [...]
}
```

Response:
```json
{
  "response": "Based on the analysis...",
  "commands": [
    {"command": "start_experiment", "description": "..."}
  ]
}
```

### POST /admin/command
Execute admin command:
```json
{
  "command": "start_experiment"
}
```

Available commands:
- `start_experiment`
- `stop_experiments`
- `pause_autonomous`
- `resume_autonomous`

## Commands the LLM Can Execute

When chatting, the LLM can suggest and execute commands by including them in its response:

**Format:** `COMMAND: {command_name}`

**Examples:**
- "COMMAND: START_EXPERIMENT" - Starts a new experiment
- "COMMAND: STOP_EXPERIMENTS" - Stops all running
- "COMMAND: PAUSE_AUTONOMOUS" - Pauses autonomous mode

## Deployment

### Initial Deployment
```bash
cd /Users/yarontorbaty/Documents/Code/AiV1
bash scripts/deploy_admin.sh
```

### Update Lambda Code Only
```bash
cd lambda
zip admin_api.zip admin_api.py
aws lambda update-function-code \
  --function-name ai-video-codec-admin-api \
  --zip-file fileb://admin_api.zip
```

### Update Frontend Only
```bash
aws s3 cp dashboard/admin.html s3://ai-video-codec-dashboard-580473065386/
aws s3 cp dashboard/admin.js s3://ai-video-codec-dashboard-580473065386/
aws cloudfront create-invalidation --distribution-id E3PUY7OMWPWSUN --paths "/*"
```

## Environment Variables

Set in Lambda console or via CLI:

```bash
aws lambda update-function-configuration \
  --function-name ai-video-codec-admin-api \
  --environment "Variables={ANTHROPIC_API_KEY=sk-ant-...}"
```

Optional (for password protection):
```bash
ADMIN_PASSWORD_HASH=$(echo -n "your-password" | sha256sum | cut -d' ' -f1)
aws lambda update-function-configuration \
  --function-name ai-video-codec-admin-api \
  --environment "Variables={ANTHROPIC_API_KEY=sk-ant-...,ADMIN_PASSWORD_HASH=$ADMIN_PASSWORD_HASH}"
```

## Monitoring

### CloudWatch Logs
```bash
aws logs tail /aws/lambda/ai-video-codec-admin-api --follow
```

### Check Lambda Status
```bash
aws lambda get-function-configuration \
  --function-name ai-video-codec-admin-api \
  --query 'State'
```

### Check API Gateway
```bash
aws apigateway get-rest-api \
  --rest-api-id mrjjwxaxma \
  --query 'name'
```

## Troubleshooting

### Chat Not Responding
1. Check Lambda logs for errors
2. Verify ANTHROPIC_API_KEY is set
3. Check API Gateway endpoint is accessible

### Commands Not Executing
1. Verify IAM role has SSM permissions
2. Check EC2 instance is running
3. Review SSM command history

### Status Not Updating
1. Check DynamoDB table exists
2. Verify Lambda has DynamoDB permissions
3. Refresh the page (cache might be stale)

## Cost Considerations

- **Lambda:** ~$0.0001 per chat request (120s timeout)
- **API Gateway:** $3.50 per million requests
- **Claude API:** ~$0.015 per message
- **DynamoDB:** Free tier covers typical usage

**Estimated monthly cost:** < $5 for moderate usage

## Future Enhancements

1. **Multi-user support:** Add user accounts and permissions
2. **Experiment history:** View past conversations and commands
3. **Visualization:** Real-time charts of experiments in admin panel
4. **Approval workflow:** Queue LLM suggestions for human approval
5. **Notifications:** Email/SMS alerts for experiment completion

## Security Best Practices

1. Set `ADMIN_PASSWORD_HASH` for production
2. Use AWS WAF to rate-limit API requests
3. Enable CloudTrail for audit logging
4. Regularly rotate the Anthropic API key
5. Monitor Lambda execution logs for anomalies

---

**Built:** October 16, 2025  
**API Endpoint:** https://mrjjwxaxma.execute-api.us-east-1.amazonaws.com/production  
**Dashboard:** https://aiv1codec.com/admin.html

