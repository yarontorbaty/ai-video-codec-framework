#!/bin/bash
# Deploy GPU Training Workers to all GPU instances

set -e

# GPU instance IDs
INSTANCES=(
    "i-08d0cb8a128aac0d6"  # training-worker 1
    "i-0c75161a102523d5c"  # training-worker 2
    "i-079a8a1e866e0badc"  # inference-worker
)

REPO_URL="https://github.com/yarontorbaty/ai-video-codec-framework.git"
TRAINING_QUEUE_URL="https://sqs.us-east-1.amazonaws.com/580473065386/ai-video-codec-training-queue"

echo "=================================================="
echo "🚀 Deploying GPU Workers"
echo "=================================================="
echo "Target instances: ${#INSTANCES[@]}"
echo "Queue: $TRAINING_QUEUE_URL"
echo ""

# Deployment commands
DEPLOY_COMMANDS=$(cat <<'EOF'
#!/bin/bash
set -e

echo "🔄 Updating GPU Training Worker..."

# Stop existing worker if running
if systemctl is-active --quiet ai-video-codec-worker 2>/dev/null; then
    echo "  Stopping existing worker..."
    sudo systemctl stop ai-video-codec-worker
fi

# Setup directory
WORK_DIR="/opt/ai-video-codec"
if [ ! -d "$WORK_DIR" ]; then
    echo "  Creating work directory..."
    sudo mkdir -p $WORK_DIR
    sudo chown ec2-user:ec2-user $WORK_DIR
fi

cd $WORK_DIR

# Clone or update repo
if [ -d ".git" ]; then
    echo "  Updating repository..."
    git fetch origin
    git reset --hard origin/main
else
    echo "  Cloning repository..."
    git clone REPO_URL_PLACEHOLDER .
fi

# Install Python dependencies
echo "  Installing dependencies..."
if command -v nvidia-smi &> /dev/null; then
    echo "  ✅ GPU detected"
    # Install PyTorch with CUDA support
    pip3 install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -5
else
    echo "  ⚠️  No GPU detected, installing CPU version"
    pip3 install --upgrade torch torchvision 2>&1 | tail -5
fi

pip3 install --upgrade boto3 2>&1 | tail -3

# Create systemd service
echo "  Creating systemd service..."
sudo tee /etc/systemd/system/ai-video-codec-worker.service > /dev/null <<SERVICE
[Unit]
Description=AI Video Codec GPU Training Worker
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/ai-video-codec
Environment="TRAINING_QUEUE_URL=QUEUE_URL_PLACEHOLDER"
Environment="PYTHONPATH=/opt/ai-video-codec"
ExecStart=/usr/bin/python3 /opt/ai-video-codec/workers/training_worker.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE

# Reload systemd and start worker
echo "  Starting worker service..."
sudo systemctl daemon-reload
sudo systemctl enable ai-video-codec-worker
sudo systemctl start ai-video-codec-worker

# Check status
sleep 2
if systemctl is-active --quiet ai-video-codec-worker; then
    echo "  ✅ Worker started successfully"
    sudo systemctl status ai-video-codec-worker --no-pager | head -10
else
    echo "  ❌ Worker failed to start"
    sudo journalctl -u ai-video-codec-worker -n 20 --no-pager
    exit 1
fi

echo "  🎉 Deployment complete!"
EOF
)

# Replace placeholders
DEPLOY_COMMANDS="${DEPLOY_COMMANDS//REPO_URL_PLACEHOLDER/$REPO_URL}"
DEPLOY_COMMANDS="${DEPLOY_COMMANDS//QUEUE_URL_PLACEHOLDER/$TRAINING_QUEUE_URL}"

# Deploy to each instance
for INSTANCE_ID in "${INSTANCES[@]}"; do
    echo ""
    echo "=================================================="
    echo "📦 Deploying to: $INSTANCE_ID"
    echo "=================================================="
    
    # Send command
    COMMAND_ID=$(aws ssm send-command \
        --instance-ids "$INSTANCE_ID" \
        --document-name "AWS-RunShellScript" \
        --parameters "commands=[\"$DEPLOY_COMMANDS\"]" \
        --output text \
        --query 'Command.CommandId')
    
    echo "  Command ID: $COMMAND_ID"
    echo "  ⏳ Waiting for deployment..."
    
    # Wait for command to complete
    sleep 5
    
    for i in {1..30}; do
        STATUS=$(aws ssm get-command-invocation \
            --command-id "$COMMAND_ID" \
            --instance-id "$INSTANCE_ID" \
            --query 'Status' \
            --output text 2>/dev/null || echo "Pending")
        
        if [ "$STATUS" = "Success" ]; then
            echo "  ✅ Deployment successful!"
            
            # Show output
            aws ssm get-command-invocation \
                --command-id "$COMMAND_ID" \
                --instance-id "$INSTANCE_ID" \
                --query 'StandardOutputContent' \
                --output text | tail -20
            break
        elif [ "$STATUS" = "Failed" ]; then
            echo "  ❌ Deployment failed!"
            aws ssm get-command-invocation \
                --command-id "$COMMAND_ID" \
                --instance-id "$INSTANCE_ID" \
                --query '[StandardOutputContent,StandardErrorContent]' \
                --output text
            break
        else
            echo "  ⏳ Status: $STATUS (attempt $i/30)"
            sleep 10
        fi
    done
done

echo ""
echo "=================================================="
echo "✅ GPU Worker Deployment Complete!"
echo "=================================================="
echo ""
echo "📊 Checking worker status..."
echo ""

# Check status on all instances
for INSTANCE_ID in "${INSTANCES[@]}"; do
    echo "Instance: $INSTANCE_ID"
    COMMAND_ID=$(aws ssm send-command \
        --instance-ids "$INSTANCE_ID" \
        --document-name "AWS-RunShellScript" \
        --parameters 'commands=["sudo systemctl status ai-video-codec-worker --no-pager | head -15"]' \
        --output text \
        --query 'Command.CommandId')
    
    sleep 3
    aws ssm get-command-invocation \
        --command-id "$COMMAND_ID" \
        --instance-id "$INSTANCE_ID" \
        --query 'StandardOutputContent' \
        --output text 2>/dev/null || echo "  Status unavailable"
    echo ""
done

echo "🎉 All workers deployed and running!"
echo ""
echo "Next steps:"
echo "  1. The LLM will automatically generate PyTorch code"
echo "  2. GPU workers will pick up jobs from SQS queue"
echo "  3. Results will appear in the dashboard"
echo ""
echo "Monitor logs:"
echo "  aws ssm send-command --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript --parameters 'commands=[\"sudo journalctl -u ai-video-codec-worker -f\"]'"

