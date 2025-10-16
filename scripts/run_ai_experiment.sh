#!/bin/bash
set -e

# AI Video Codec Framework - Run AI Experiment Script
# This script runs the AI codec experiment with procedural generation

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Video Codec - Run Experiment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Configuration
PROJECT_NAME="ai-video-codec"
ENVIRONMENT="production"
REGION="us-east-1"
EXPERIMENT_ID="exp_$(date +%s)"

echo -e "${BLUE}Experiment ID: ${EXPERIMENT_ID}${NC}"
echo -e "${BLUE}Timestamp: $(date)${NC}"
echo ""

# Check if Python environment is set up
echo -e "${BLUE}Checking Python environment...${NC}"
if ! python3 -c "import torch, cv2, numpy, boto3" 2>/dev/null; then
    echo -e "${YELLOW}Installing required Python packages...${NC}"
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip3 install opencv-python numpy boto3 pyyaml psutil
fi
echo -e "${GREEN}✓ Python environment ready${NC}"

# Check if AWS CLI is configured
echo -e "${BLUE}Checking AWS configuration...${NC}"
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}Error: AWS CLI not configured${NC}"
    echo "Please run: ./scripts/setup_aws.sh"
    exit 1
fi
echo -e "${GREEN}✓ AWS CLI configured${NC}"

# Check if test data exists in S3
echo -e "${BLUE}Checking test data availability...${NC}"
VIDEOS_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name ${PROJECT_NAME}-${ENVIRONMENT}-storage \
    --query 'Stacks[0].Outputs[?OutputKey==`VideosBucketName`].OutputValue' \
    --output text \
    --region ${REGION})

if [ "$VIDEOS_BUCKET" = "None" ] || [ -z "$VIDEOS_BUCKET" ]; then
    echo -e "${RED}Error: Could not get videos bucket name${NC}"
    exit 1
fi

# Check if test videos exist
SOURCE_EXISTS=$(aws s3 ls s3://$VIDEOS_BUCKET/source/SOURCE_HD_RAW.mp4 2>/dev/null | wc -l)
HEVC_EXISTS=$(aws s3 ls s3://$VIDEOS_BUCKET/hevc/HEVC_HD_10Mbps.mp4 2>/dev/null | wc -l)

if [ "$SOURCE_EXISTS" -eq 0 ] || [ "$HEVC_EXISTS" -eq 0 ]; then
    echo -e "${YELLOW}Test data not found. Uploading test data...${NC}"
    ./scripts/upload_test_data.sh
fi
echo -e "${GREEN}✓ Test data available${NC}"

# Create necessary directories
echo -e "${BLUE}Setting up directories...${NC}"
mkdir -p data/{source,hevc,compressed,results}
mkdir -p logs models checkpoints
echo -e "${GREEN}✓ Directories created${NC}"

# Run the AI codec experiment
echo -e "${BLUE}Starting AI codec experiment...${NC}"
echo -e "${YELLOW}This may take 1-2 hours depending on system performance${NC}"
echo ""

# Set timeout for the experiment (2 hours)
timeout 7200 python3 -m src.agents.experiment_orchestrator \
    --config config/ai_codec_config.yaml \
    --experiment-id "$EXPERIMENT_ID" \
    --action start

EXPERIMENT_EXIT_CODE=$?

if [ $EXPERIMENT_EXIT_CODE -eq 124 ]; then
    echo -e "${RED}✗ Experiment timed out after 2 hours${NC}"
    exit 1
elif [ $EXPERIMENT_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}✗ Experiment failed with exit code $EXPERIMENT_EXIT_CODE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ AI codec experiment completed successfully${NC}"

# Check results
echo -e "${BLUE}Checking experiment results...${NC}"
RESULTS_FILE="data/results/${EXPERIMENT_ID}_results.json"

if [ -f "$RESULTS_FILE" ]; then
    echo -e "${GREEN}✓ Results file created: $RESULTS_FILE${NC}"
    
    # Display key metrics
    echo -e "${BLUE}Key Results:${NC}"
    python3 -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    results = json.load(f)
    
print('Experiment ID:', results.get('experiment_id', 'N/A'))
print('Timestamp:', results.get('timestamp', 'N/A'))

# AI Codec Results
ai_results = results.get('ai_codec_results', {})
if 'ai_codec_metrics' in ai_results:
    metrics = ai_results['ai_codec_metrics']
    print('\\nAI Codec Metrics:')
    print('  Bitrate:', f\"{metrics.get('bitrate_mbps', 0):.2f} Mbps\")
    print('  PSNR:', f\"{metrics.get('psnr_db', 0):.2f} dB\")
    print('  Compression Ratio:', f\"{metrics.get('compression_ratio', 0):.4f}\")

# Hybrid Results
hybrid_results = results.get('hybrid_results', {})
if 'bitrate_mbps' in hybrid_results:
    print('\\nHybrid Results:')
    print('  Bitrate:', f\"{hybrid_results.get('bitrate_mbps', 0):.2f} Mbps\")
    print('  PSNR:', f\"{hybrid_results.get('psnr_db', 0):.2f} dB\")
    print('  Compression Ratio:', f\"{hybrid_results.get('compression_ratio', 0):.4f}\")

# Comparison
comparison = results.get('comparison', {})
targets = comparison.get('target_achievement', {})
print('\\nTarget Achievement:')
print('  Bitrate Target (< 1 Mbps):', '✓' if targets.get('bitrate_target', False) else '✗')
print('  PSNR Target (> 35 dB):', '✓' if targets.get('psnr_target', False) else '✗')
print('  Compression Target (< 0.1):', '✓' if targets.get('compression_target', False) else '✗')
"
    
else
    echo -e "${RED}✗ Results file not found: $RESULTS_FILE${NC}"
fi

# Upload results to S3
echo -e "${BLUE}Uploading results to S3...${NC}"
if [ -f "$RESULTS_FILE" ]; then
    aws s3 cp "$RESULTS_FILE" "s3://$VIDEOS_BUCKET/results/" \
        --region "$REGION" \
        --metadata "experiment_id=$EXPERIMENT_ID,type=experiment_results"
    echo -e "${GREEN}✓ Results uploaded to S3${NC}"
fi

# Check resource usage
echo -e "${BLUE}Resource Usage Summary:${NC}"
if [ -f "$RESULTS_FILE" ]; then
    python3 -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    results = json.load(f)
    
resource_usage = results.get('resource_usage', {})
if resource_usage:
    print('  Average CPU:', f\"{resource_usage.get('avg_cpu_percent', 0):.1f}%\")
    print('  Max CPU:', f\"{resource_usage.get('max_cpu_percent', 0):.1f}%\")
    print('  Average Memory:', f\"{resource_usage.get('avg_memory_percent', 0):.1f}%\")
    print('  Max Memory:', f\"{resource_usage.get('max_memory_percent', 0):.1f}%\")
    if resource_usage.get('gpu_usage'):
        print('  GPU Usage:', f\"{resource_usage.get('gpu_usage', 0):.1f}%\")
"

# Cost information
echo -e "${BLUE}Cost Information:${NC}"
if [ -f "$RESULTS_FILE" ]; then
    python3 -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    results = json.load(f)
    
cost = results.get('cost', {})
if cost:
    print('  Current Cost: $' + str(cost.get('current_cost', 0)))
    print('  Daily Cost: $' + str(cost.get('daily_cost', 0)))
"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}AI Codec Experiment Completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${BLUE}Experiment Details:${NC}"
echo -e "${GREEN}Experiment ID:${NC} $EXPERIMENT_ID"
echo -e "${GREEN}Results File:${NC} $RESULTS_FILE"
echo -e "${GREEN}S3 Location:${NC} s3://$VIDEOS_BUCKET/results/"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Check the dashboard for real-time metrics"
echo "2. Review detailed results in $RESULTS_FILE"
echo "3. Compare with HEVC baseline (10 Mbps target)"
echo "4. Analyze which approach (AI, Procedural, Hybrid) performed best"
echo ""
echo -e "${GREEN}AI Codec Framework is now running! 🎬🚀${NC}"
