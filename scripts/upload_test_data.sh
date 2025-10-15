#!/bin/bash
set -e

# AI Video Codec Framework - Test Data Upload Script
# This script uploads test videos to S3 for the framework to process

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="ai-video-codec"
ENVIRONMENT="production"
REGION="us-east-1"
STACK_NAME="${PROJECT_NAME}-${ENVIRONMENT}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Video Codec Framework - Test Data Upload${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}Error: AWS CLI not configured${NC}"
    exit 1
fi

# Get S3 bucket names
echo -e "${BLUE}Getting S3 bucket information...${NC}"
ARTIFACTS_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME}-storage \
    --query 'Stacks[0].Outputs[?OutputKey==`ArtifactsBucketName`].OutputValue' \
    --output text \
    --region ${REGION} 2>/dev/null)

VIDEOS_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME}-storage \
    --query 'Stacks[0].Outputs[?OutputKey==`VideosBucketName`].OutputValue' \
    --output text \
    --region ${REGION} 2>/dev/null)

if [ "$ARTIFACTS_BUCKET" = "None" ] || [ -z "$ARTIFACTS_BUCKET" ]; then
    echo -e "${RED}Error: Could not get artifacts bucket. Is the storage stack deployed?${NC}"
    exit 1
fi

if [ "$VIDEOS_BUCKET" = "None" ] || [ -z "$VIDEOS_BUCKET" ]; then
    echo -e "${RED}Error: Could not get videos bucket. Is the storage stack deployed?${NC}"
    exit 1
fi

echo -e "${GREEN}Artifacts Bucket: ${ARTIFACTS_BUCKET}${NC}"
echo -e "${GREEN}Videos Bucket: ${VIDEOS_BUCKET}${NC}"
echo ""

# Check if test data directory exists
if [ ! -d "data" ]; then
    echo -e "${YELLOW}Creating data directory...${NC}"
    mkdir -p data
fi

# Check if test videos exist
if [ ! -f "data/source_4k60_10s.mp4" ]; then
    echo -e "${YELLOW}Test video not found: data/source_4k60_10s.mp4${NC}"
    echo "Please provide a 4K60 10-second test video"
    echo "You can:"
    echo "1. Download a sample video"
    echo "2. Create your own test video"
    echo "3. Use an existing video file"
    echo ""
    read -p "Enter path to your test video (or press Enter to skip): " VIDEO_PATH
    
    if [ -n "$VIDEO_PATH" ] && [ -f "$VIDEO_PATH" ]; then
        echo -e "${BLUE}Copying video to data directory...${NC}"
        cp "$VIDEO_PATH" data/source_4k60_10s.mp4
        echo -e "${GREEN}✓ Video copied${NC}"
    else
        echo -e "${YELLOW}Skipping video upload${NC}"
        exit 0
    fi
fi

# Check if HEVC reference exists
if [ ! -f "data/hevc_reference.mp4" ]; then
    echo -e "${YELLOW}HEVC reference not found: data/hevc_reference.mp4${NC}"
    echo "Creating HEVC reference from source video..."
    
    if command -v ffmpeg &> /dev/null; then
        echo -e "${BLUE}Encoding HEVC reference...${NC}"
        ffmpeg -i data/source_4k60_10s.mp4 -c:v libx265 -preset medium -crf 23 -c:a copy data/hevc_reference.mp4 -y
        echo -e "${GREEN}✓ HEVC reference created${NC}"
    else
        echo -e "${RED}Error: FFmpeg not found. Please install FFmpeg to create HEVC reference${NC}"
        echo "You can install it with: brew install ffmpeg (macOS) or apt install ffmpeg (Ubuntu)"
        exit 1
    fi
fi

# Upload source video
echo -e "${BLUE}Uploading source video...${NC}"
aws s3 cp data/source_4k60_10s.mp4 s3://${VIDEOS_BUCKET}/source.mp4 \
    --region ${REGION} \
    --metadata "type=source,resolution=4k,fps=60,duration=10s"

echo -e "${GREEN}✓ Source video uploaded${NC}"

# Upload HEVC reference
echo -e "${BLUE}Uploading HEVC reference...${NC}"
aws s3 cp data/hevc_reference.mp4 s3://${VIDEOS_BUCKET}/hevc_reference.mp4 \
    --region ${REGION} \
    --metadata "type=hevc_reference,codec=hevc,preset=medium,crf=23"

echo -e "${GREEN}✓ HEVC reference uploaded${NC}"

# Create metadata file
echo -e "${BLUE}Creating metadata file...${NC}"
cat > data/metadata.json << EOF
{
  "source_video": {
    "filename": "source.mp4",
    "resolution": "4K",
    "fps": 60,
    "duration": 10,
    "codec": "uncompressed",
    "description": "Original uncompressed 4K60 10-second test video"
  },
  "hevc_reference": {
    "filename": "hevc_reference.mp4",
    "codec": "hevc",
    "preset": "medium",
    "crf": 23,
    "description": "HEVC-encoded reference for comparison"
  },
  "upload_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "framework_version": "0.1.0"
}
EOF

# Upload metadata
aws s3 cp data/metadata.json s3://${ARTIFACTS_BUCKET}/metadata.json \
    --region ${REGION} \
    --content-type "application/json"

echo -e "${GREEN}✓ Metadata uploaded${NC}"

# Create experiment configuration
echo -e "${BLUE}Creating experiment configuration...${NC}"
cat > data/experiment_config.json << EOF
{
  "experiment_id": "baseline_test_$(date +%Y%m%d_%H%M%S)",
  "name": "Baseline Test Experiment",
  "description": "Initial baseline test with provided video",
  "source_video": "s3://${VIDEOS_BUCKET}/source.mp4",
  "hevc_reference": "s3://${VIDEOS_BUCKET}/hevc_reference.mp4",
  "target_compression": 0.90,
  "target_psnr": 95.0,
  "max_training_time": 7200,
  "max_epochs": 100,
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "pending"
}
EOF

# Upload experiment configuration
aws s3 cp data/experiment_config.json s3://${ARTIFACTS_BUCKET}/experiments/baseline_test.json \
    --region ${REGION} \
    --content-type "application/json"

echo -e "${GREEN}✓ Experiment configuration uploaded${NC}"

# Show file sizes
echo -e "${BLUE}File sizes:${NC}"
if [ -f "data/source_4k60_10s.mp4" ]; then
    SOURCE_SIZE=$(du -h data/source_4k60_10s.mp4 | cut -f1)
    echo -e "${GREEN}Source video: ${SOURCE_SIZE}${NC}"
fi

if [ -f "data/hevc_reference.mp4" ]; then
    HEVC_SIZE=$(du -h data/hevc_reference.mp4 | cut -f1)
    echo -e "${GREEN}HEVC reference: ${HEVC_SIZE}${NC}"
fi

# Calculate compression ratio
if [ -f "data/source_4k60_10s.mp4" ] && [ -f "data/hevc_reference.mp4" ]; then
    SOURCE_BYTES=$(stat -f%z data/source_4k60_10s.mp4 2>/dev/null || stat -c%s data/source_4k60_10s.mp4 2>/dev/null)
    HEVC_BYTES=$(stat -f%z data/hevc_reference.mp4 2>/dev/null || stat -c%s data/hevc_reference.mp4 2>/dev/null)
    
    if [ -n "$SOURCE_BYTES" ] && [ -n "$HEVC_BYTES" ] && [ "$SOURCE_BYTES" -gt 0 ]; then
        COMPRESSION_RATIO=$(echo "scale=2; $HEVC_BYTES * 100 / $SOURCE_BYTES" | bc -l 2>/dev/null || echo "N/A")
        echo -e "${GREEN}HEVC compression ratio: ${COMPRESSION_RATIO}%${NC}"
    fi
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Test data upload complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${BLUE}Uploaded files:${NC}"
echo "- Source video: s3://${VIDEOS_BUCKET}/source.mp4"
echo "- HEVC reference: s3://${VIDEOS_BUCKET}/hevc_reference.mp4"
echo "- Metadata: s3://${ARTIFACTS_BUCKET}/metadata.json"
echo "- Experiment config: s3://${ARTIFACTS_BUCKET}/experiments/baseline_test.json"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Check orchestrator status: ./scripts/monitor_aws.sh"
echo "2. SSH to orchestrator: ssh ec2-user@\$(aws cloudformation describe-stacks --stack-name ${STACK_NAME}-compute --query 'Stacks[0].Outputs[?OutputKey==\`OrchestratorPublicIP\`].OutputValue' --output text --region ${REGION})"
echo "3. View orchestrator logs: sudo journalctl -u ai-video-codec-orchestrator -f"
echo "4. Monitor experiment progress: ./scripts/monitor_aws.sh --watch"
echo ""
echo -e "${GREEN}Ready to start experiments! 🚀${NC}"
