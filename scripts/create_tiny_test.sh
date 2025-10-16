#!/bin/bash
set -e

# AI Video Codec Framework - Tiny Test Video Creation
# Creates very small test videos for quick testing

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Video Codec - Tiny Test Video Creator${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create test data directory
mkdir -p test_data

echo -e "${BLUE}Creating tiny but challenging test videos...${NC}"

# 1. Create a tiny source video (720p, 2 seconds, 30fps) - very fast
echo -e "${BLUE}1. Creating tiny source video (720p, 2s, 30fps)...${NC}"
timeout 30 ffmpeg -f lavfi -i testsrc2=size=1280x720:rate=30:duration=2 \
    -f lavfi -i testsrc2=size=1280x720:rate=30:duration=2 \
    -filter_complex "
        [0:v]scale=1280x720,format=yuv420p[base];
        [1:v]scale=640x360,format=yuv420p,scale=1280x720[overlay];
        [base][overlay]blend=all_mode=multiply:all_opacity=0.5,
        noise=alls=20:allf=t,
        unsharp=5:5:0.8:3:3:0.4
    " \
    -c:v libx264 -preset ultrafast -crf 0 -pix_fmt yuv420p \
    -y test_data/source_tiny.mp4

if [ $? -eq 124 ]; then
    echo -e "${RED}✗ Video creation timed out after 30 seconds${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Tiny source video created${NC}"

# 2. Create HEVC reference (same content, HEVC encoded)
echo -e "${BLUE}2. Creating HEVC reference...${NC}"
timeout 60 ffmpeg -i test_data/source_tiny.mp4 \
    -c:v libx265 -crf 23 -preset fast -pix_fmt yuv420p \
    -y test_data/hevc_tiny.mp4

if [ $? -eq 124 ]; then
    echo -e "${RED}✗ HEVC creation timed out after 60 seconds${NC}"
    exit 1
fi

echo -e "${GREEN}✓ HEVC reference created${NC}"

# Get file sizes for comparison
SOURCE_SIZE=$(stat -f%z test_data/source_tiny.mp4 2>/dev/null || stat -c%s test_data/source_tiny.mp4 2>/dev/null)
HEVC_SIZE=$(stat -f%z test_data/hevc_tiny.mp4 2>/dev/null || stat -c%s test_data/hevc_tiny.mp4 2>/dev/null)

# Convert to MB
SOURCE_SIZE_MB=$((SOURCE_SIZE / 1024 / 1024))
HEVC_SIZE_MB=$((HEVC_SIZE / 1024 / 1024))

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Tiny Test Videos Created Successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${BLUE}Generated Files:${NC}"
echo -e "${GREEN}Source (720p, 2s):${NC} test_data/source_tiny.mp4 (${SOURCE_SIZE_MB} MB)"
echo -e "${GREEN}HEVC Reference:${NC} test_data/hevc_tiny.mp4 (${HEVC_SIZE_MB} MB)"
echo ""
echo -e "${BLUE}Video Characteristics:${NC}"
echo "• Resolution: 720p (1280x720)"
echo "• Frame Rate: 30 FPS"
echo "• Duration: 2 seconds"
echo "• Content: Geometric test patterns with noise"
echo "• Challenge: High-frequency details, contrast"
echo ""
echo -e "${BLUE}Timeout Protection:${NC}"
echo "• Source creation: 30 second timeout"
echo "• HEVC creation: 60 second timeout"
echo "• Operations will fail fast if they take too long"
echo ""
echo -e "${GREEN}Ready for quick upload to S3! 🎬🚀${NC}"
