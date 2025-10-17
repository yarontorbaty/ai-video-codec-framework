#!/bin/bash
# Setup script for GPU worker (runs on GPU instance)
set -e

echo "Updating GPU Training Worker..."

# Stop existing worker if running
if systemctl is-active --quiet ai-video-codec-worker 2>/dev/null; then
    echo "Stopping existing worker..."
    sudo systemctl stop ai-video-codec-worker
fi

# Setup directory
WORK_DIR="/opt/ai-video-codec"
if [ ! -d "$WORK_DIR" ]; then
    echo "Creating work directory..."
    sudo mkdir -p $WORK_DIR
    sudo chown ec2-user:ec2-user $WORK_DIR
fi

cd $WORK_DIR

# Clone or update repo
if [ -d ".git" ]; then
    echo "Updating repository..."
    sudo git config --global --add safe.directory /opt/ai-video-codec
    git fetch origin
    git reset --hard origin/main
else
    echo "Cloning repository..."
    git clone https://github.com/yarontorbaty/ai-video-codec-framework.git .
fi

# Install Python dependencies
echo "Installing dependencies..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected - installing PyTorch with CUDA"
    pip3 install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "No GPU detected - installing CPU version"
    pip3 install --upgrade torch torchvision
fi

pip3 install --upgrade boto3

# Create systemd service
echo "Creating systemd service..."
sudo tee /etc/systemd/system/ai-video-codec-worker.service > /dev/null <<'SERVICE'
[Unit]
Description=AI Video Codec GPU Training Worker
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/ai-video-codec
Environment="TRAINING_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/580473065386/ai-video-codec-training-queue"
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
echo "Starting worker service..."
sudo systemctl daemon-reload
sudo systemctl enable ai-video-codec-worker
sudo systemctl start ai-video-codec-worker

# Check status
sleep 2
if systemctl is-active --quiet ai-video-codec-worker; then
    echo "Worker started successfully"
    sudo systemctl status ai-video-codec-worker --no-pager | head -10
else
    echo "Worker failed to start"
    sudo journalctl -u ai-video-codec-worker -n 20 --no-pager
    exit 1
fi

echo "Deployment complete!"

