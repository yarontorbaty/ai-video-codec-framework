#!/bin/bash
set -e

# AI Video Codec Framework - Quick Test Video Creation with FFmpeg
# Creates challenging test videos using FFmpeg's built-in generators

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Video Codec - Quick Test Video Creator${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create test data directory
mkdir -p test_data

echo -e "${BLUE}Creating challenging test videos with FFmpeg...${NC}"

# 1. Create a source video with complex patterns (4K, 10 seconds, 60fps)
echo -e "${BLUE}1. Creating source video (4K, 10s, 60fps)...${NC}"
ffmpeg -f lavfi -i testsrc2=size=3840x2160:rate=60:duration=10 \
    -f lavfi -i testsrc2=size=3840x2160:rate=60:duration=10 \
    -filter_complex "
        [0:v]scale=3840x2160,format=yuv420p[base];
        [1:v]scale=1920x1080,format=yuv420p,scale=3840x2160[overlay];
        [base][overlay]blend=all_mode=multiply:all_opacity=0.5,
        noise=alls=20:allf=t,
        curves=preset=strong_contrast,
        unsharp=5:5:0.8:3:3:0.4
    " \
    -c:v libx264 -preset ultrafast -crf 0 -pix_fmt yuv420p \
    -y test_data/source_4k60_10s.mp4

echo -e "${GREEN}✓ Source video created${NC}"

# 2. Create HEVC reference (same content, HEVC encoded)
echo -e "${BLUE}2. Creating HEVC reference...${NC}"
ffmpeg -i test_data/source_4k60_10s.mp4 \
    -c:v libx265 -crf 23 -preset medium -pix_fmt yuv420p \
    -y test_data/hevc_4k60_10s.mp4

echo -e "${GREEN}✓ HEVC reference created${NC}"

# 3. Create additional challenging test patterns
echo -e "${BLUE}3. Creating additional test patterns...${NC}"

# High-frequency noise pattern
ffmpeg -f lavfi -i testsrc2=size=3840x2160:rate=60:duration=5 \
    -f lavfi -i noise=alls=100:allf=t:size=3840x2160:rate=60:duration=5 \
    -filter_complex "
        [0:v][1:v]blend=all_mode=multiply:all_opacity=0.8,
        unsharp=5:5:1.0:3:3:0.6,
        curves=preset=strong_contrast
    " \
    -c:v libx264 -preset ultrafast -crf 0 -pix_fmt yuv420p \
    -y test_data/source_noise_4k60_5s.mp4

# Complex motion pattern
ffmpeg -f lavfi -i testsrc2=size=3840x2160:rate=60:duration=5 \
    -f lavfi -i testsrc2=size=3840x2160:rate=60:duration=5 \
    -filter_complex "
        [0:v]scale=3840x2160,format=yuv420p[base];
        [1:v]scale=1920x1080,format=yuv420p,scale=3840x2160[overlay];
        [base][overlay]blend=all_mode=screen:all_opacity=0.7,
        rotate=PI/180*sin(t*2*PI/10),
        unsharp=5:5:0.8:3:3:0.4
    " \
    -c:v libx264 -preset ultrafast -crf 0 -pix_fmt yuv420p \
    -y test_data/source_motion_4k60_5s.mp4

echo -e "${GREEN}✓ Additional test patterns created${NC}"

# Get file sizes
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
echo -e "${GREEN}Source (4K, 10s):${NC} test_data/source_4k60_10s.mp4 (${SOURCE_SIZE_MB} MB)"
echo -e "${GREEN}HEVC Reference:${NC} test_data/hevc_4k60_10s.mp4 (${HEVC_SIZE_MB} MB)"
echo -e "${GREEN}Noise Pattern:${NC} test_data/source_noise_4k60_5s.mp4"
echo -e "${GREEN}Motion Pattern:${NC} test_data/source_motion_4k60_5s.mp4"
echo ""
echo -e "${BLUE}Video Characteristics:${NC}"
echo "• Resolution: 4K (3840x2160)"
echo "• Frame Rate: 60 FPS"
echo "• Duration: 10 seconds (main), 5 seconds (patterns)"
echo "• Content: Complex geometric patterns with noise"
echo "• Challenge: High-frequency details, motion, contrast"
echo ""
echo -e "${BLUE}FFmpeg Commands Used:${NC}"
echo "• testsrc2: Geometric test patterns"
echo "• noise: High-frequency noise"
echo "• blend: Complex overlays"
echo "• unsharp: Sharpening filters"
echo "• curves: Contrast enhancement"
echo "• rotate: Motion effects"
echo ""
echo -e "${GREEN}Ready for upload to S3! 🎬🚀${NC}"
