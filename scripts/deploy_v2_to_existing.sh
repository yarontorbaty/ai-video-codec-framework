#!/bin/bash
#
# Deploy v2.0 Code to Existing Infrastructure
# Uses existing EC2 instances and deploys new v2.0 components
#

set -e

echo "=========================================="
echo "🚀 DEPLOYING v2.0 TO EXISTING INFRASTRUCTURE"
echo "=========================================="
echo ""

# Get instance IPs from AWS
ORCHESTRATOR_IP="34.239.1.29"
GPU_WORKER_1="18.208.180.67"
GPU_WORKER_2="3.92.194.242"
GPU_WORKER_3="3.231.221.71"
GPU_WORKER_4="184.72.95.161"

# SSH key (assumes default location)
SSH_KEY="${HOME}/.ssh/ai-video-codec-key.pem"
if [ ! -f "$SSH_KEY" ]; then
    SSH_KEY="${HOME}/.ssh/ai-codec-key.pem"
fi
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found. Please specify key location:"
    read -p "SSH key path: " SSH_KEY
fi

SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

echo "📋 Deployment Plan:"
echo "  Orchestrator: $ORCHESTRATOR_IP"
echo "  GPU Workers: 4 instances"
echo "  SSH Key: $SSH_KEY"
echo ""

# Function to deploy to a host
deploy_to_host() {
    local HOST=$1
    local TYPE=$2
    
    echo "=========================================="
    echo "📦 Deploying to $TYPE: $HOST"
    echo "=========================================="
    
    # Upload new v2.0 files
    echo "  📤 Uploading v2.0 files..."
    
    if [ "$TYPE" = "orchestrator" ]; then
        # Upload orchestrator files
        scp $SSH_OPTS src/agents/gpu_first_orchestrator.py ubuntu@$HOST:~/ai-video-codec-framework/src/agents/
        scp $SSH_OPTS src/agents/encoding_agent.py ubuntu@$HOST:~/ai-video-codec-framework/src/agents/
        scp $SSH_OPTS src/agents/decoding_agent.py ubuntu@$HOST:~/ai-video-codec-framework/src/agents/
        scp $SSH_OPTS requirements.txt ubuntu@$HOST:~/ai-video-codec-framework/
        
        echo "  ✅ Orchestrator files uploaded"
        
    elif [ "$TYPE" = "gpu_worker" ]; then
        # Upload GPU worker files
        scp $SSH_OPTS workers/neural_codec_gpu_worker.py ubuntu@$HOST:~/ai-video-codec-framework/workers/
        scp $SSH_OPTS src/agents/encoding_agent.py ubuntu@$HOST:~/ai-video-codec-framework/src/agents/
        scp $SSH_OPTS src/agents/decoding_agent.py ubuntu@$HOST:~/ai-video-codec-framework/src/agents/
        scp $SSH_OPTS requirements.txt ubuntu@$HOST:~/ai-video-codec-framework/
        
        echo "  ✅ GPU worker files uploaded"
    fi
    
    # Install new dependencies
    echo "  📦 Installing dependencies..."
    ssh $SSH_OPTS ubuntu@$HOST << 'ENDSSH'
cd ~/ai-video-codec-framework
source venv/bin/activate 2>/dev/null || python3 -m venv venv && source venv/bin/activate
pip3 install -q scikit-image thop
echo "  ✅ Dependencies installed"
ENDSSH
    
    echo "  ✅ Deployment to $HOST complete"
    echo ""
}

# Deploy to orchestrator
deploy_to_host "$ORCHESTRATOR_IP" "orchestrator"

# Deploy to all GPU workers
for GPU_IP in $GPU_WORKER_1 $GPU_WORKER_2 $GPU_WORKER_3 $GPU_WORKER_4; do
    deploy_to_host "$GPU_IP" "gpu_worker"
done

echo "=========================================="
echo "✅ DEPLOYMENT COMPLETE"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start GPU Worker (pick one):"
echo "   ssh $SSH_OPTS ubuntu@$GPU_WORKER_1"
echo "   cd ai-video-codec-framework && source venv/bin/activate"
echo "   python3 workers/neural_codec_gpu_worker.py"
echo ""
echo "2. Start Orchestrator:"
echo "   ssh $SSH_OPTS ubuntu@$ORCHESTRATOR_IP"
echo "   cd ai-video-codec-framework && source venv/bin/activate"
echo "   export ANTHROPIC_API_KEY=sk-ant-..."
echo "   python3 src/agents/gpu_first_orchestrator.py"
echo ""

