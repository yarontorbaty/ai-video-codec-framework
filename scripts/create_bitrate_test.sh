#!/bin/bash
set -e

# AI Video Codec Framework - Bitrate-Targeted Test Video Creation
# Creates test videos with specific HEVC bitrate targets for comparison

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Video Codec - Bitrate-Targeted Test Creator${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create test data directory
mkdir -p test_data

echo -e "${BLUE}Creating test videos with specific bitrate targets...${NC}"

# 1. Create 1080p source video (2 seconds, 30fps) - fast generation
echo -e "${BLUE}1. Creating 1080p source video (2s, 30fps)...${NC}"
timeout 30 ffmpeg -f lavfi -i testsrc2=size=1920x1080:rate=30:duration=2 \
    -f lavfi -i testsrc2=size=1920x1080:rate=30:duration=2 \
    -filter_complex "
        [0:v]scale=1920x1080,format=yuv420p[base];
        [1:v]scale=960x540,format=yuv420p,scale=1920x1080[overlay];
        [base][overlay]blend=all_mode=multiply:all_opacity=0.5,
        noise=alls=20:allf=t,
        unsharp=5:5:0.8:3:3:0.4
    " \
    -c:v libx264 -preset ultrafast -crf 0 -pix_fmt yuv420p \
    -y test_data/source_1080p.mp4

if [ $? -eq 124 ]; then
    echo -e "${RED}✗ 1080p source creation timed out after 30 seconds${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 1080p source video created${NC}"

# 2. Create 4K source video (1 second, 30fps) - smaller for faster processing
echo -e "${BLUE}2. Creating 4K source video (1s, 30fps)...${NC}"
timeout 45 ffmpeg -f lavfi -i testsrc2=size=3840x2160:rate=30:duration=1 \
    -f lavfi -i testsrc2=size=3840x2160:rate=30:duration=1 \
    -filter_complex "
        [0:v]scale=3840x2160,format=yuv420p[base];
        [1:v]scale=1920x1080,format=yuv420p,scale=3840x2160[overlay];
        [base][overlay]blend=all_mode=multiply:all_opacity=0.5,
        noise=alls=20:allf=t,
        unsharp=5:5:0.8:3:3:0.4
    " \
    -c:v libx264 -preset ultrafast -crf 0 -pix_fmt yuv420p \
    -y test_data/source_4k.mp4

if [ $? -eq 124 ]; then
    echo -e "${RED}✗ 4K source creation timed out after 45 seconds${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 4K source video created${NC}"

# 3. Create HEVC reference at 10 Mbps for 1080p
echo -e "${BLUE}3. Creating HEVC reference at 10 Mbps for 1080p...${NC}"
timeout 60 ffmpeg -i test_data/source_1080p.mp4 \
    -c:v libx265 -b:v 10M -maxrate 10M -bufsize 20M -preset fast -pix_fmt yuv420p \
    -y test_data/hevc_1080p_10mbps.mp4

if [ $? -eq 124 ]; then
    echo -e "${RED}✗ 1080p HEVC creation timed out after 60 seconds${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 1080p HEVC reference created (10 Mbps target)${NC}"

# 4. Create HEVC reference at 25 Mbps for 4K
echo -e "${BLUE}4. Creating HEVC reference at 25 Mbps for 4K...${NC}"
timeout 90 ffmpeg -i test_data/source_4k.mp4 \
    -c:v libx265 -b:v 25M -maxrate 25M -bufsize 50M -preset fast -pix_fmt yuv420p \
    -y test_data/hevc_4k_25mbps.mp4

if [ $? -eq 124 ]; then
    echo -e "${RED}✗ 4K HEVC creation timed out after 90 seconds${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 4K HEVC reference created (25 Mbps target)${NC}"

# 5. Validate all videos with ffprobe
echo -e "${BLUE}5. Validating videos with ffprobe...${NC}"

validate_video() {
    local file="$1"
    local expected_resolution="$2"
    local expected_bitrate="$3"
    
    echo -e "${BLUE}Validating: $(basename "$file")${NC}"
    
    # Check if file exists and is readable
    if [ ! -f "$file" ]; then
        echo -e "${RED}✗ File not found: $file${NC}"
        return 1
    fi
    
    # Get video info
    local info=$(ffprobe -v quiet -print_format json -show_streams "$file" 2>/dev/null)
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ ffprobe failed for: $file${NC}"
        return 1
    fi
    
    # Extract resolution
    local width=$(echo "$info" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['streams'][0]['width'])" 2>/dev/null)
    local height=$(echo "$info" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['streams'][0]['height'])" 2>/dev/null)
    local bitrate=$(echo "$info" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['streams'][0].get('bit_rate', 'N/A'))" 2>/dev/null)
    local codec=$(echo "$info" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['streams'][0]['codec_name'])" 2>/dev/null)
    
    echo -e "${GREEN}  Resolution: ${width}x${height}${NC}"
    echo -e "${GREEN}  Codec: ${codec}${NC}"
    echo -e "${GREEN}  Bitrate: ${bitrate} bps${NC}"
    
    # Validate resolution
    if [ "$expected_resolution" = "1080p" ] && [ "$width" = "1920" ] && [ "$height" = "1080" ]; then
        echo -e "${GREEN}  ✓ Resolution correct for 1080p${NC}"
    elif [ "$expected_resolution" = "4K" ] && [ "$width" = "3840" ] && [ "$height" = "2160" ]; then
        echo -e "${GREEN}  ✓ Resolution correct for 4K${NC}"
    else
        echo -e "${RED}  ✗ Resolution mismatch${NC}"
        return 1
    fi
    
    # Validate bitrate (allow some tolerance)
    if [ "$bitrate" != "N/A" ] && [ "$expected_bitrate" != "N/A" ]; then
        local bitrate_mbps=$((bitrate / 1000000))
        local expected_mbps=$((expected_bitrate / 1000000))
        local tolerance=2
        
        if [ $((bitrate_mbps - expected_mbps)) -le $tolerance ] && [ $((expected_mbps - bitrate_mbps)) -le $tolerance ]; then
            echo -e "${GREEN}  ✓ Bitrate within tolerance (${bitrate_mbps} Mbps vs ${expected_mbps} Mbps target)${NC}"
        else
            echo -e "${YELLOW}  ⚠ Bitrate outside tolerance (${bitrate_mbps} Mbps vs ${expected_mbps} Mbps target)${NC}"
        fi
    fi
    
    echo -e "${GREEN}  ✓ Video is decodable${NC}"
    return 0
}

# Validate all videos
validate_video "test_data/source_1080p.mp4" "1080p" "N/A"
validate_video "test_data/source_4k.mp4" "4K" "N/A"
validate_video "test_data/hevc_1080p_10mbps.mp4" "1080p" "10000000"
validate_video "test_data/hevc_4k_25mbps.mp4" "4K" "25000000"

# Get file sizes for comparison
SOURCE_1080P_SIZE=$(stat -f%z test_data/source_1080p.mp4 2>/dev/null || stat -c%s test_data/source_1080p.mp4 2>/dev/null)
SOURCE_4K_SIZE=$(stat -f%z test_data/source_4k.mp4 2>/dev/null || stat -c%s test_data/source_4k.mp4 2>/dev/null)
HEVC_1080P_SIZE=$(stat -f%z test_data/hevc_1080p_10mbps.mp4 2>/dev/null || stat -c%s test_data/hevc_1080p_10mbps.mp4 2>/dev/null)
HEVC_4K_SIZE=$(stat -f%z test_data/hevc_4k_25mbps.mp4 2>/dev/null || stat -c%s test_data/hevc_4k_25mbps.mp4 2>/dev/null)

# Convert to MB
SOURCE_1080P_SIZE_MB=$((SOURCE_1080P_SIZE / 1024 / 1024))
SOURCE_4K_SIZE_MB=$((SOURCE_4K_SIZE / 1024 / 1024))
HEVC_1080P_SIZE_MB=$((HEVC_1080P_SIZE / 1024 / 1024))
HEVC_4K_SIZE_MB=$((HEVC_4K_SIZE / 1024 / 1024))

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Bitrate-Targeted Test Videos Created!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${BLUE}Generated Files:${NC}"
echo -e "${GREEN}1080p Source:${NC} test_data/source_1080p.mp4 (${SOURCE_1080P_SIZE_MB} MB)"
echo -e "${GREEN}1080p HEVC (10 Mbps):${NC} test_data/hevc_1080p_10mbps.mp4 (${HEVC_1080P_SIZE_MB} MB)"
echo -e "${GREEN}4K Source:${NC} test_data/source_4k.mp4 (${SOURCE_4K_SIZE_MB} MB)"
echo -e "${GREEN}4K HEVC (25 Mbps):${NC} test_data/hevc_4k_25mbps.mp4 (${HEVC_4K_SIZE_MB} MB)"
echo ""
echo -e "${BLUE}Compression Targets:${NC}"
echo -e "${GREEN}1080p:${NC} 10 Mbps HEVC baseline (AI codec must beat this)"
echo -e "${GREEN}4K:${NC} 25 Mbps HEVC baseline (AI codec must beat this)"
echo ""
echo -e "${BLUE}AI Codec Goals:${NC}"
echo -e "${GREEN}1080p:${NC} < 1 Mbps (90% reduction from 10 Mbps)"
echo -e "${GREEN}4K:${NC} < 2.5 Mbps (90% reduction from 25 Mbps)"
echo -e "${GREEN}PSNR:${NC} > 95% (maintain quality)"
echo ""
echo -e "${BLUE}Timeout Protection:${NC}"
echo "• 1080p source: 30s timeout"
echo "• 4K source: 45s timeout"
echo "• 1080p HEVC: 60s timeout"
echo "• 4K HEVC: 90s timeout"
echo "• All operations fail fast if they take too long"
echo ""
echo -e "${GREEN}Ready for AI codec testing! 🎬🚀${NC}"
