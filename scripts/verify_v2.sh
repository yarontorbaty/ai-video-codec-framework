#!/bin/bash
#
# Verification Script: v2.0 GPU-First Neural Codec
# Checks that all components are working correctly
#
# Usage: ./verify_v2.sh
#

set -e  # Exit on error

echo "=========================================="
echo "🔍 VERIFICATION: v2.0 System Health"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_WARNING=0

check_pass() {
    echo -e "${GREEN}✅ $1${NC}"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
}

check_fail() {
    echo -e "${RED}❌ $1${NC}"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
}

check_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    CHECKS_WARNING=$((CHECKS_WARNING + 1))
}

echo -e "${BLUE}1. File Structure Verification${NC}"
echo ""

# Check v2.0 files exist
if [ -f "src/agents/encoding_agent.py" ]; then
    check_pass "encoding_agent.py exists"
else
    check_fail "encoding_agent.py missing"
fi

if [ -f "src/agents/decoding_agent.py" ]; then
    check_pass "decoding_agent.py exists"
else
    check_fail "decoding_agent.py missing"
fi

if [ -f "src/agents/gpu_first_orchestrator.py" ]; then
    check_pass "gpu_first_orchestrator.py exists"
else
    check_fail "gpu_first_orchestrator.py missing"
fi

if [ -f "workers/neural_codec_gpu_worker.py" ]; then
    check_pass "neural_codec_gpu_worker.py exists"
else
    check_fail "workers/neural_codec_gpu_worker.py missing"
fi

# Check documentation
if [ -f "GPU_NEURAL_CODEC_ARCHITECTURE.md" ]; then
    check_pass "Architecture documentation exists"
else
    check_warn "Architecture documentation missing"
fi

echo ""
echo -e "${BLUE}2. Python Environment Verification${NC}"
echo ""

# Check venv
if [ -d "venv" ]; then
    check_pass "Virtual environment exists"
    source venv/bin/activate
else
    check_fail "Virtual environment missing"
fi

# Check Python packages
python3 -c "import torch" 2>/dev/null && check_pass "PyTorch installed" || check_warn "PyTorch not installed (needed for GPU worker)"
python3 -c "import cv2" 2>/dev/null && check_pass "OpenCV installed" || check_fail "OpenCV not installed"
python3 -c "import boto3" 2>/dev/null && check_pass "Boto3 installed" || check_fail "Boto3 not installed"
python3 -c "import numpy" 2>/dev/null && check_pass "NumPy installed" || check_fail "NumPy not installed"

echo ""
echo -e "${BLUE}3. AWS Resources Verification${NC}"
echo ""

# Check AWS credentials
if aws sts get-caller-identity > /dev/null 2>&1; then
    check_pass "AWS credentials configured"
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    echo "   Account: $ACCOUNT_ID"
else
    check_fail "AWS credentials not configured"
fi

# Check SQS queue
QUEUE_URL="https://sqs.us-east-1.amazonaws.com/580473065386/ai-video-codec-training-queue"
if aws sqs get-queue-attributes --queue-url "$QUEUE_URL" --attribute-names ApproximateNumberOfMessages > /dev/null 2>&1; then
    check_pass "SQS queue accessible"
    QUEUE_MSGS=$(aws sqs get-queue-attributes --queue-url "$QUEUE_URL" --attribute-names ApproximateNumberOfMessages --query 'Attributes.ApproximateNumberOfMessages' --output text)
    echo "   Messages in queue: $QUEUE_MSGS"
else
    check_fail "SQS queue not accessible"
fi

# Check DynamoDB table
if aws dynamodb describe-table --table-name ai-video-codec-experiments > /dev/null 2>&1; then
    check_pass "DynamoDB table accessible"
    ITEM_COUNT=$(aws dynamodb scan --table-name ai-video-codec-experiments --select COUNT --query 'Count' --output text 2>/dev/null || echo "?")
    echo "   Experiments in table: $ITEM_COUNT"
else
    check_fail "DynamoDB table not accessible"
fi

# Check S3 bucket
BUCKET="ai-video-codec-videos-${ACCOUNT_ID}"
if aws s3 ls "s3://${BUCKET}/" > /dev/null 2>&1; then
    check_pass "S3 bucket accessible"
    echo "   Bucket: s3://${BUCKET}"
else
    check_warn "S3 bucket not accessible (may need to create)"
fi

echo ""
echo -e "${BLUE}4. Process Verification${NC}"
echo ""

# Check if GPU worker is running (on this machine - may not be)
GPU_WORKER_PID=$(pgrep -f "neural_codec_gpu_worker.py" 2>/dev/null || echo "")
if [ -n "$GPU_WORKER_PID" ]; then
    check_pass "GPU worker running (PID: $GPU_WORKER_PID)"
else
    check_warn "GPU worker not running on this machine"
    echo "   (Should be running on GPU worker instance)"
fi

# Check if orchestrator is running
ORCH_PID=$(pgrep -f "gpu_first_orchestrator.py" 2>/dev/null || echo "")
if [ -n "$ORCH_PID" ]; then
    check_pass "Orchestrator running (PID: $ORCH_PID)"
else
    check_warn "Orchestrator not running"
    echo "   (Start with: python3 src/agents/gpu_first_orchestrator.py)"
fi

# Check if old v1.0 services are still running
OLD_ORCH_PID=$(pgrep -f "procedural_experiment_runner.py" 2>/dev/null || echo "")
if [ -n "$OLD_ORCH_PID" ]; then
    check_warn "OLD v1.0 orchestrator still running (PID: $OLD_ORCH_PID)"
    echo "   Stop it with: kill $OLD_ORCH_PID"
else
    check_pass "No v1.0 orchestrator running"
fi

echo ""
echo -e "${BLUE}5. LLM Configuration Verification${NC}"
echo ""

if [ -n "$ANTHROPIC_API_KEY" ]; then
    check_pass "Anthropic API key set"
elif [ -n "$OPENAI_API_KEY" ]; then
    check_pass "OpenAI API key set"
else
    check_fail "No LLM API key set"
    echo "   Set with: export ANTHROPIC_API_KEY=sk-ant-..."
fi

echo ""
echo -e "${BLUE}6. Recent Experiments Check${NC}"
echo ""

# Check for recent experiments
RECENT_COUNT=$(aws dynamodb scan \
    --table-name ai-video-codec-experiments \
    --select COUNT \
    --filter-expression "begins_with(experiment_id, :prefix)" \
    --expression-attribute-values '{":prefix":{"S":"gpu_exp_"}}' \
    --query 'Count' \
    --output text 2>/dev/null || echo "0")

if [ "$RECENT_COUNT" -gt 0 ]; then
    check_pass "Found $RECENT_COUNT v2.0 experiments"
    
    # Get most recent experiment
    LATEST=$(aws dynamodb scan \
        --table-name ai-video-codec-experiments \
        --filter-expression "begins_with(experiment_id, :prefix)" \
        --expression-attribute-values '{":prefix":{"S":"gpu_exp_"}}' \
        --query 'Items | sort_by(@, &timestamp.N) | [-1]' \
        --output json 2>/dev/null)
    
    if [ -n "$LATEST" ] && [ "$LATEST" != "null" ]; then
        echo "   Latest experiment:"
        echo "$LATEST" | python3 -c "import sys, json; data = json.load(sys.stdin); print(f\"     ID: {data.get('experiment_id', {}).get('S', 'N/A')}\"); print(f\"     Status: {data.get('status', {}).get('S', 'N/A')}\"); gpu_results = data.get('gpu_results', {}).get('M', {}); bitrate = gpu_results.get('bitrate_mbps', {}).get('N', 'N/A'); psnr = gpu_results.get('psnr_db', {}).get('N', 'N/A'); print(f\"     Bitrate: {bitrate} Mbps\"); print(f\"     PSNR: {psnr} dB\") if gpu_results else None" 2>/dev/null || echo "     (parsing failed)"
    fi
else
    check_warn "No v2.0 experiments yet"
    echo "   (Run orchestrator to start experiments)"
fi

echo ""
echo "=========================================="
echo -e "${BLUE}VERIFICATION SUMMARY${NC}"
echo "=========================================="
echo ""
echo -e "${GREEN}✅ Passed: $CHECKS_PASSED${NC}"
echo -e "${RED}❌ Failed: $CHECKS_FAILED${NC}"
echo -e "${YELLOW}⚠️  Warnings: $CHECKS_WARNING${NC}"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 System is healthy!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Ensure GPU worker is running on GPU instance"
    echo "2. Start orchestrator: python3 src/agents/gpu_first_orchestrator.py"
    echo "3. Monitor logs and DynamoDB for results"
    exit 0
else
    echo -e "${RED}⚠️  System has issues that need attention${NC}"
    echo ""
    echo "Fix the failed checks above before proceeding."
    exit 1
fi


