#!/bin/bash
set -e

ORCH_ID="i-00d8ebe7d25026fdd"
WORKER_ID="i-01113a08e8005b235"
WORKER_IP="172.31.73.149"
BUCKET="ai-codec-v3-artifacts-580473065386"

echo "📤 Deploying Worker Code"
echo "========================"

aws ssm send-command \
  --instance-ids $WORKER_ID \
  --document-name "AWS-RunShellScript" \
  --parameters commands=["
    sudo yum install -y python3 python3-pip &&
    cd /opt &&
    sudo aws s3 cp s3://$BUCKET/deploy/worker.tar.gz . &&
    sudo tar -xzf worker.tar.gz &&
    cd worker &&
    sudo pip3 install -r requirements.txt &&
    export S3_BUCKET=$BUCKET &&
    export AWS_REGION=us-east-1 &&
    sudo nohup python3 main.py > /var/log/worker.log 2>&1 &
    echo 'Worker deployed and started'
  "] \
  --output text \
  --query 'Command.CommandId'

echo "✅ Worker deployment initiated"
echo ""

echo "📤 Deploying Orchestrator Code"
echo "==============================="

aws ssm send-command \
  --instance-ids $ORCH_ID \
  --document-name "AWS-RunShellScript" \
  --parameters commands=["
    sudo yum install -y python3 python3-pip &&
    cd /opt &&
    sudo aws s3 cp s3://$BUCKET/deploy/orchestrator.tar.gz . &&
    sudo tar -xzf orchestrator.tar.gz &&
    cd orchestrator &&
    sudo pip3 install -r requirements.txt &&
    export WORKER_URL=http://$WORKER_IP:8080 &&
    export DYNAMODB_TABLE=ai-codec-v3-experiments &&
    export AWS_REGION=us-east-1 &&
    export MAX_ITERATIONS=10 &&
    sudo nohup python3 main.py > /var/log/orchestrator.log 2>&1 &
    echo 'Orchestrator deployed and started'
  "] \
  --output text \
  --query 'Command.CommandId'

echo "✅ Orchestrator deployment initiated"
echo ""
echo "⏳ Waiting 2 minutes for services to start..."
sleep 120

echo "✅ Deployment complete!"
echo ""
echo "Check logs with:"
echo "  aws ssm start-session --target $WORKER_ID"
echo "  sudo tail -f /var/log/worker.log"
echo ""
echo "  aws ssm start-session --target $ORCH_ID"
echo "  sudo tail -f /var/log/orchestrator.log"
