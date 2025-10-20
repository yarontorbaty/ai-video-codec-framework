#!/bin/bash

ORCHESTRATOR_INSTANCE="i-0ee283400d2e131a4"
WORKER_IP="172.31.65.58"
REGION="us-east-1"

echo "🚀 Deploying Fixed Evolutionary Orchestrator"
echo "============================================="

# 1. Upload to S3
echo "1️⃣ Uploading fixed orchestrator to S3..."
aws s3 cp ./v3/orchestrator/evolutionary_orchestrator.py \
    s3://ai-codec-v3-artifacts-580473065386/code/evo_orchestrator_FIXED.py \
    --region $REGION

# 2. Stop old orchestrator
echo "2️⃣ Stopping old orchestrator..."
aws ssm send-command \
    --instance-ids $ORCHESTRATOR_INSTANCE \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["pkill -f evolutionary_orchestrator || true"]' \
    --region $REGION

sleep 5

# 3. Deploy and start new orchestrator
echo "3️⃣ Deploying and starting fixed orchestrator..."
COMMAND_ID=$(aws ssm send-command \
    --instance-ids $ORCHESTRATOR_INSTANCE \
    --document-name "AWS-RunShellScript" \
    --parameters "commands=[\"cd /home/ec2-user/evolutionary-orchestrator\",\"aws s3 cp s3://ai-codec-v3-artifacts-580473065386/code/evo_orchestrator_FIXED.py ./evolutionary_orchestrator.py --region $REGION\",\"echo 'Verifying fix:'\",\"grep -c '_performance_score' evolutionary_orchestrator.py\",\"grep -c 'Score:' evolutionary_orchestrator.py\",\"pkill -f evolutionary || true\",\"nohup python3 evolutionary_orchestrator.py http://$WORKER_IP:8080 50 > evo_run_fixed.log 2>&1 &\",\"sleep 3\",\"echo 'Started! Checking logs:'\",\"tail -20 evo_run_fixed.log\"]" \
    --region $REGION \
    --query 'Command.CommandId' \
    --output text)

echo "   Command ID: $COMMAND_ID"
echo "   Waiting 15 seconds..."
sleep 15

# 4. Check deployment status
echo "4️⃣ Checking deployment status..."
aws ssm get-command-invocation \
    --command-id $COMMAND_ID \
    --instance-id $ORCHESTRATOR_INSTANCE \
    --region $REGION \
    --query '[Status,StandardOutputContent]' \
    --output text

echo ""
echo "✅ Deployment complete!"
echo ""
echo "To monitor logs:"
echo "aws ssm send-command --instance-ids $ORCHESTRATOR_INSTANCE --document-name 'AWS-RunShellScript' --parameters 'commands=[\"tail -f /home/ec2-user/evolutionary-orchestrator/evo_run_fixed.log\"]' --region $REGION"

