#!/bin/bash
set -e

# AI Video Codec Framework - Test Data Upload Script
# This script uploads test videos to S3 for the framework

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

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Video Codec Framework - Test Data Upload${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}Error: AWS CLI not configured${NC}"
    exit 1
fi

echo -e "${GREEN}✓ AWS CLI configured${NC}"

# Get S3 bucket names from CloudFormation
echo -e "${BLUE}Getting S3 bucket names...${NC}"
VIDEOS_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name ${PROJECT_NAME}-${ENVIRONMENT}-storage \
    --query 'Stacks[0].Outputs[?OutputKey==`VideosBucketName`].OutputValue' \
    --output text \
    --region ${REGION})

if [ "$VIDEOS_BUCKET" = "None" ] || [ -z "$VIDEOS_BUCKET" ]; then
    echo -e "${RED}Error: Could not get videos bucket name${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Videos bucket: ${VIDEOS_BUCKET}${NC}"

# Check if test data exists
TEST_DATA_DIR="test_data"
if [ ! -d "$TEST_DATA_DIR" ]; then
    echo -e "${YELLOW}Test data directory not found. Creating test videos...${NC}"
    
    # Check if Python and OpenCV are available
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python3 not found${NC}"
        exit 1
    fi
    
    # Install required packages
    echo -e "${BLUE}Installing required packages...${NC}"
    pip3 install opencv-python numpy --user
    
    # Generate test videos
    echo -e "${BLUE}Generating test videos...${NC}"
    python3 scripts/generate_test_videos.py \
        --output-dir "$TEST_DATA_DIR" \
        --width 3840 \
        --height 2160 \
        --duration 10 \
        --fps 60 \
        --patterns mixed natural \
        --create-hevc \
        --hevc-crf 23
fi

# Check if test videos exist
SOURCE_VIDEO=""
HEVC_VIDEO=""

# Look for HD test videos first
if [ -f "$TEST_DATA_DIR/SOURCE_HD_RAW.mp4" ]; then
    SOURCE_VIDEO="$TEST_DATA_DIR/SOURCE_HD_RAW.mp4"
    HEVC_VIDEO="$TEST_DATA_DIR/HEVC_HD_10Mbps.mp4"
    echo -e "${GREEN}✓ Found HD test videos${NC}"
elif [ -f "$TEST_DATA_DIR/source_tiny.mp4" ]; then
    SOURCE_VIDEO="$TEST_DATA_DIR/source_tiny.mp4"
    HEVC_VIDEO="$TEST_DATA_DIR/hevc_tiny.mp4"
    echo -e "${GREEN}✓ Found tiny test videos${NC}"
else
    # Look for source videos (legacy)
    for video in "$TEST_DATA_DIR"/source_4k60_10s_*.mp4; do
        if [ -f "$video" ]; then
            SOURCE_VIDEO="$video"
            break
        fi
    done

    # Look for HEVC videos (legacy)
    for video in "$TEST_DATA_DIR"/hevc_4k60_10s_*.mp4; do
        if [ -f "$video" ]; then
            HEVC_VIDEO="$video"
            break
        fi
    done
fi

if [ -z "$SOURCE_VIDEO" ]; then
    echo -e "${RED}Error: No source test video found in $TEST_DATA_DIR${NC}"
    echo "Please run: ./scripts/create_hd_test.sh"
    exit 1
fi

if [ -z "$HEVC_VIDEO" ]; then
    echo -e "${YELLOW}Warning: No HEVC reference video found${NC}"
    echo "Creating HEVC reference..."
    
    # Check if FFmpeg is available
    if ! command -v ffmpeg &> /dev/null; then
        echo -e "${RED}Error: FFmpeg not found. Please install FFmpeg to create HEVC reference${NC}"
        echo "On macOS: brew install ffmpeg"
        echo "On Ubuntu: sudo apt install ffmpeg"
        exit 1
    fi
    
    HEVC_VIDEO="$TEST_DATA_DIR/hevc_4k60_10s_reference.mp4"
    ffmpeg -i "$SOURCE_VIDEO" -c:v libx265 -crf 23 -preset medium -pix_fmt yuv420p -y "$HEVC_VIDEO"
    echo -e "${GREEN}✓ HEVC reference created${NC}"
fi

echo -e "${BLUE}Uploading test videos to S3...${NC}"

# Upload source video
echo -e "${BLUE}Uploading source video: $(basename "$SOURCE_VIDEO")${NC}"
if [[ "$SOURCE_VIDEO" == *"SOURCE_HD_RAW"* ]]; then
    aws s3 cp "$SOURCE_VIDEO" "s3://$VIDEOS_BUCKET/source/" \
        --region "$REGION" \
        --metadata "type=source,resolution=1080p,fps=30,duration=10s"
else
    aws s3 cp "$SOURCE_VIDEO" "s3://$VIDEOS_BUCKET/source/" \
        --region "$REGION" \
        --metadata "type=source,resolution=4k,fps=60,duration=10s"
fi

# Upload HEVC reference
echo -e "${BLUE}Uploading HEVC reference: $(basename "$HEVC_VIDEO")${NC}"
if [[ "$HEVC_VIDEO" == *"HEVC_HD_10Mbps"* ]]; then
    aws s3 cp "$HEVC_VIDEO" "s3://$VIDEOS_BUCKET/hevc/" \
        --region "$REGION" \
        --metadata "type=hevc,resolution=1080p,fps=30,duration=10s,bitrate=10mbps"
else
    aws s3 cp "$HEVC_VIDEO" "s3://$VIDEOS_BUCKET/hevc/" \
        --region "$REGION" \
        --metadata "type=hevc,resolution=4k,fps=60,duration=10s,crf=23"
fi

# Get file sizes for comparison
SOURCE_SIZE=$(stat -f%z "$SOURCE_VIDEO" 2>/dev/null || stat -c%s "$SOURCE_VIDEO" 2>/dev/null)
HEVC_SIZE=$(stat -f%z "$HEVC_VIDEO" 2>/dev/null || stat -c%s "$HEVC_VIDEO" 2>/dev/null)

# Convert to MB
SOURCE_SIZE_MB=$((SOURCE_SIZE / 1024 / 1024))
HEVC_SIZE_MB=$((HEVC_SIZE / 1024 / 1024))

echo -e "${GREEN}✓ Test data upload complete${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Test Data Upload Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${BLUE}Uploaded Files:${NC}"
echo -e "${GREEN}Source Video:${NC} $(basename "$SOURCE_VIDEO") (${SOURCE_SIZE_MB} MB)"
echo -e "${GREEN}HEVC Reference:${NC} $(basename "$HEVC_VIDEO") (${HEVC_SIZE_MB} MB)"
echo ""
echo -e "${BLUE}S3 Locations:${NC}"
echo -e "${GREEN}Source:${NC} s3://$VIDEOS_BUCKET/source/"
echo -e "${GREEN}HEVC:${NC} s3://$VIDEOS_BUCKET/hevc/"
echo ""
echo -e "${BLUE}Video Characteristics:${NC}"
if [[ "$SOURCE_VIDEO" == *"SOURCE_HD_RAW"* ]]; then
    echo "• Resolution: 1080p (1920x1080)"
    echo "• Frame Rate: 30 FPS"
    echo "• Duration: 10 seconds"
    echo "• Pattern: Geometric test patterns with noise"
    echo "• Challenge: High-frequency details, contrast, motion"
else
    echo "• Resolution: 4K (3840x2160)"
    echo "• Frame Rate: 60 FPS"
    echo "• Duration: 10 seconds"
    echo "• Pattern: Complex mixed patterns"
    echo "• Challenge: High-frequency details, motion, textures"
fi
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Start an experiment using the uploaded test data"
echo "2. Monitor progress on the dashboard"
echo "3. Compare AI codec results vs HEVC reference"
echo ""
echo -e "${GREEN}Ready for AI codec testing! 🎬🚀${NC}"