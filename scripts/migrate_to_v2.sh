#!/bin/bash
#
# Migration Script: v1.0 → v2.0 GPU-First Neural Codec
# This script helps migrate from the old system to the new GPU-first architecture
#
# Usage: ./migrate_to_v2.sh
#

set -e  # Exit on error

echo "=========================================="
echo "🔄 MIGRATION: v1.0 → v2.0"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running on orchestrator
if [ ! -d "/home/ubuntu/ai-video-codec-framework" ]; then
    echo -e "${YELLOW}⚠️  Not running from standard path${NC}"
    echo "Current directory: $(pwd)"
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

echo -e "${BLUE}Step 1: Verify Git Repository${NC}"
echo "Checking for new v2.0 files..."

# Check if new files exist
if [ ! -f "src/agents/encoding_agent.py" ]; then
    echo -e "${RED}❌ src/agents/encoding_agent.py not found${NC}"
    echo "Please pull latest code: git pull origin main"
    exit 1
fi

if [ ! -f "src/agents/decoding_agent.py" ]; then
    echo -e "${RED}❌ src/agents/decoding_agent.py not found${NC}"
    exit 1
fi

if [ ! -f "src/agents/gpu_first_orchestrator.py" ]; then
    echo -e "${RED}❌ src/agents/gpu_first_orchestrator.py not found${NC}"
    exit 1
fi

if [ ! -f "workers/neural_codec_gpu_worker.py" ]; then
    echo -e "${RED}❌ workers/neural_codec_gpu_worker.py not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All v2.0 files present${NC}"
echo ""

echo -e "${BLUE}Step 2: Stop v1.0 Services${NC}"

# Check if old orchestrator is running
OLD_PID=$(pgrep -f "procedural_experiment_runner.py" || echo "")
if [ -n "$OLD_PID" ]; then
    echo "Found old orchestrator running (PID: $OLD_PID)"
    echo "Stop it? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        kill $OLD_PID
        echo -e "${GREEN}✅ Stopped old orchestrator${NC}"
    else
        echo -e "${YELLOW}⚠️  Old orchestrator still running${NC}"
    fi
else
    echo -e "${GREEN}✅ No old orchestrator running${NC}"
fi

# Check systemd service
if systemctl is-active --quiet neural-codec-orchestrator 2>/dev/null; then
    echo "Found systemd service running"
    echo "Stop it? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        sudo systemctl stop neural-codec-orchestrator
        echo -e "${GREEN}✅ Stopped systemd service${NC}"
    fi
else
    echo -e "${GREEN}✅ No systemd service running${NC}"
fi

echo ""

echo -e "${BLUE}Step 3: Verify Python Dependencies${NC}"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found${NC}"
    echo "Create it? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        python3 -m venv venv
        echo -e "${GREEN}✅ Created virtual environment${NC}"
    fi
fi

# Activate venv
source venv/bin/activate

# Check required packages
echo "Checking dependencies..."
python3 -c "import torch" 2>/dev/null || {
    echo -e "${YELLOW}⚠️  PyTorch not installed${NC}"
    echo "Note: GPU worker needs PyTorch with CUDA"
}

python3 -c "import cv2" 2>/dev/null || {
    echo -e "${RED}❌ OpenCV not installed${NC}"
    echo "Install: pip install opencv-python"
    exit 1
}

python3 -c "import boto3" 2>/dev/null || {
    echo -e "${RED}❌ Boto3 not installed${NC}"
    echo "Install: pip install boto3"
    exit 1
}

echo -e "${GREEN}✅ Dependencies OK${NC}"
echo ""

echo -e "${BLUE}Step 4: Verify AWS Configuration${NC}"

# Check AWS credentials
aws sts get-caller-identity > /dev/null 2>&1 || {
    echo -e "${RED}❌ AWS credentials not configured${NC}"
    echo "Run: aws configure"
    exit 1
}

echo "AWS Identity:"
aws sts get-caller-identity --output table

# Check SQS queue
echo ""
echo "Checking SQS queue..."
QUEUE_URL="https://sqs.us-east-1.amazonaws.com/580473065386/ai-video-codec-training-queue"
aws sqs get-queue-attributes \
    --queue-url "$QUEUE_URL" \
    --attribute-names ApproximateNumberOfMessages \
    --output table 2>/dev/null || {
    echo -e "${RED}❌ Cannot access SQS queue${NC}"
    exit 1
}

echo -e "${GREEN}✅ SQS access OK${NC}"

# Check DynamoDB table
echo ""
echo "Checking DynamoDB table..."
aws dynamodb describe-table \
    --table-name ai-video-codec-experiments \
    --output table 2>/dev/null || {
    echo -e "${RED}❌ Cannot access DynamoDB table${NC}"
    exit 1
}

echo -e "${GREEN}✅ DynamoDB access OK${NC}"

# Check S3 bucket
echo ""
echo "Checking S3 bucket..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET="ai-video-codec-videos-${ACCOUNT_ID}"
aws s3 ls "s3://${BUCKET}/" > /dev/null 2>&1 || {
    echo -e "${YELLOW}⚠️  Cannot access S3 bucket: ${BUCKET}${NC}"
    echo "You may need to upload test video later"
}

echo -e "${GREEN}✅ S3 access OK${NC}"
echo ""

echo -e "${BLUE}Step 5: Check LLM API Key${NC}"

if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  No LLM API key set${NC}"
    echo "Set one of:"
    echo "  export ANTHROPIC_API_KEY=sk-ant-..."
    echo "  export OPENAI_API_KEY=sk-..."
    echo ""
    echo "Set it now? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Enter your API key:"
        read -r api_key
        export ANTHROPIC_API_KEY="$api_key"
        echo -e "${GREEN}✅ API key set${NC}"
    else
        echo -e "${YELLOW}⚠️  No API key - orchestrator will fail${NC}"
    fi
else
    echo -e "${GREEN}✅ LLM API key configured${NC}"
fi

echo ""

echo "=========================================="
echo -e "${GREEN}✅ MIGRATION PREPARATION COMPLETE${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. On GPU Worker Instance:"
echo "   ssh ubuntu@<gpu-worker-ip>"
echo "   cd ai-video-codec-framework"
echo "   source venv/bin/activate"
echo "   python3 workers/neural_codec_gpu_worker.py"
echo ""
echo "2. On Orchestrator Instance (this machine):"
echo "   source venv/bin/activate"
echo "   export ANTHROPIC_API_KEY=sk-ant-..."
echo "   python3 src/agents/gpu_first_orchestrator.py"
echo ""
echo "3. Verify with:"
echo "   ./scripts/verify_v2.sh"
echo ""
echo -e "${BLUE}Ready to start v2.0!${NC}"


