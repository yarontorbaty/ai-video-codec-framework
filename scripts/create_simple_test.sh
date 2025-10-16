#!/bin/bash
set -e

# AI Video Codec Framework - Simple Test Video Creation
# Creates basic but challenging test videos quickly

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Video Codec - Simple Test Video Creator${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create test data directory
mkdir -p test_data

echo -e "${BLUE}Creating simple but challenging test videos...${NC}"

# 1. Create a simple source video (1080p, 5 seconds, 30fps) - much faster
echo -e "${BLUE}1. Creating source video (1080p, 5s, 30fps)...${NC}"
ffmpeg -f lavfi -i testsrc2=size=1920x1080:rate=30:duration=5 \
    -f lavfi -i testsrc2=size=1920x1080:rate=30:duration=5 \
    -filter_complex "
        [0:v]scale=1920x1080,format=yuv420p[base];
        [1:v]scale=960x540,format=yuv420p,scale=1920x1080[overlay];
        [base][overlay]blend=all_mode=multiply:all_opacity=0.5,
        noise=alls=20:allf=t,
        unsharp=5:5:0.8:3:3:0.4
    " \
    -c:v libx264 -preset ultrafast -crf 0 -pix_fmt yuv420p \
    -y test_data/source_4k60_10s.mp4

echo -e "${GREEN}✓ Source video created${NC}"

# 2. Create HEVC reference (same content, HEVC encoded)
echo -e "${BLUE}2. Creating HEVC reference...${NC}"
ffmpeg -i test_data/source_4k60_10s.mp4 \
    -c:v libx265 -crf 23 -preset fast -pix_fmt yuv420p \
    -y test_data/hevc_4k60_10s.mp4

echo -e "${GREEN}✓ HEVC reference created${NC}"

# Get file sizes for comparison
SOURCE_SIZE=$(stat -f%z test_data/source_4k60_10s.mp4 2>/dev/null || stat -c%s test_data/source_4k60_10s.mp4 2>/dev/null)
HEVC_SIZE=$(stat -f%z test_data/hevc_4k60_10s.mp4 2>/dev/null || stat -c%s test_data/hevc_4k60_10s.mp4 2>/dev/null)

# Convert to MB
SOURCE_SIZE_MB=$((SOURCE_SIZE / 1024 / 1024))
HEVC_SIZE_MB=$((HEVC_SIZE / 1024 / 1024))

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Test Videos Created Successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${BLUE}Generated Files:${NC}"
echo -e "${GREEN}Source (1080p, 5s):${NC} test_data/source_4k60_10s.mp4 (${SOURCE_SIZE_MB} MB)"
echo -e "${GREEN}HEVC Reference:${NC} test_data/hevc_4k60_10s.mp4 (${HEVC_SIZE_MB} MB)"
echo ""
echo -e "${BLUE}Video Characteristics:${NC}"
echo "• Resolution: 1080p (1920x1080)"
echo "• Frame Rate: 30 FPS"
echo "• Duration: 5 seconds"
echo "• Content: Geometric test patterns with noise"
echo "• Challenge: High-frequency details, contrast"
echo ""
echo -e "${BLUE}FFmpeg Commands Used:${NC}"
echo "• testsrc2: Geometric test patterns"
echo "• noise: High-frequency noise"
echo "• blend: Complex overlays"
echo "• unsharp: Sharpening filters"
echo ""
echo -e "${GREEN}Ready for upload to S3! 🎬🚀${NC}"
