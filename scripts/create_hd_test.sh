#!/bin/bash
set -e

# AI Video Codec Framework - HD Test Video Creation
# Creates simpler HD test videos: SOURCE HD RAW, HEVC HD 10Mbps
# All clips are 10 seconds long, easier to work with

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Video Codec - HD Test Video Creator${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create test data directory
mkdir -p test_data

echo -e "${BLUE}Creating HD test videos (10 seconds each)...${NC}"

# 1. Create SOURCE HD RAW (10 seconds, 1080p, 30fps) - simpler content
echo -e "${BLUE}1. Creating SOURCE HD RAW (10s, 1080p, 30fps)...${NC}"
timeout 60 ffmpeg -f lavfi -i testsrc2=size=1920x1080:rate=30:duration=10 \
    -f lavfi -i testsrc2=size=1920x1080:rate=30:duration=10 \
    -filter_complex "
        [0:v]scale=1920x1080,format=yuv420p[base];
        [1:v]scale=960x540,format=yuv420p,scale=1920x1080[overlay];
        [base][overlay]blend=all_mode=multiply:all_opacity=0.3,
        noise=alls=10:allf=t,
        unsharp=5:5:0.8:3:3:0.4
    " \
    -c:v libx264 -preset ultrafast -crf 0 -pix_fmt yuv420p \
    -y test_data/SOURCE_HD_RAW.mp4

if [ $? -eq 124 ]; then
    echo -e "${RED}✗ SOURCE HD RAW creation timed out after 60 seconds${NC}"
    exit 1
fi

echo -e "${GREEN}✓ SOURCE HD RAW created${NC}"

# 2. Create HEVC HD at 10 Mbps (10 seconds)
echo -e "${BLUE}2. Creating HEVC HD 10Mbps (10s)...${NC}"
timeout 120 ffmpeg -i test_data/SOURCE_HD_RAW.mp4 \
    -c:v libx265 -b:v 10M -maxrate 10M -bufsize 20M -preset fast -pix_fmt yuv420p \
    -y test_data/HEVC_HD_10Mbps.mp4

if [ $? -eq 124 ]; then
    echo -e "${RED}✗ HEVC HD creation timed out after 120 seconds${NC}"
    exit 1
fi

echo -e "${GREEN}✓ HEVC HD 10Mbps created${NC}"

# 3. Validate all videos with ffprobe
echo -e "${BLUE}3. Validating videos with ffprobe...${NC}"

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
    local duration=$(echo "$info" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['streams'][0]['duration'])" 2>/dev/null)
    
    echo -e "${GREEN}  Resolution: ${width}x${height}${NC}"
    echo -e "${GREEN}  Codec: ${codec}${NC}"
    echo -e "${GREEN}  Duration: ${duration}s${NC}"
    echo -e "${GREEN}  Bitrate: ${bitrate} bps${NC}"
    
    # Validate resolution
    if [ "$expected_resolution" = "1080p" ] && [ "$width" = "1920" ] && [ "$height" = "1080" ]; then
        echo -e "${GREEN}  ✓ Resolution correct for 1080p${NC}"
    else
        echo -e "${RED}  ✗ Resolution mismatch${NC}"
        return 1
    fi
    
    # Validate duration (should be ~10 seconds)
    if [ "$duration" != "N/A" ]; then
        local duration_float=$(echo "$duration" | python3 -c "import sys; print(float(sys.stdin.read().strip()))" 2>/dev/null)
        if [ $(echo "$duration_float >= 9.5 && $duration_float <= 10.5" | bc -l) -eq 1 ]; then
            echo -e "${GREEN}  ✓ Duration correct (~10s)${NC}"
        else
            echo -e "${YELLOW}  ⚠ Duration outside expected range (${duration}s)${NC}"
        fi
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
validate_video "test_data/SOURCE_HD_RAW.mp4" "1080p" "N/A"
validate_video "test_data/HEVC_HD_10Mbps.mp4" "1080p" "10000000"

# Get file sizes for comparison
SOURCE_SIZE=$(stat -f%z test_data/SOURCE_HD_RAW.mp4 2>/dev/null || stat -c%s test_data/SOURCE_HD_RAW.mp4 2>/dev/null)
HEVC_SIZE=$(stat -f%z test_data/HEVC_HD_10Mbps.mp4 2>/dev/null || stat -c%s test_data/HEVC_HD_10Mbps.mp4 2>/dev/null)

# Convert to MB
SOURCE_SIZE_MB=$((SOURCE_SIZE / 1024 / 1024))
HEVC_SIZE_MB=$((HEVC_SIZE / 1024 / 1024))

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}HD Test Videos Created Successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${BLUE}Generated Files (10 seconds each):${NC}"
echo -e "${GREEN}SOURCE HD RAW:${NC} test_data/SOURCE_HD_RAW.mp4 (${SOURCE_SIZE_MB} MB)"
echo -e "${GREEN}HEVC HD 10Mbps:${NC} test_data/HEVC_HD_10Mbps.mp4 (${HEVC_SIZE_MB} MB)"
echo ""
echo -e "${BLUE}AI Codec Challenge:${NC}"
echo -e "${GREEN}HD Target:${NC} Beat 10 Mbps HEVC (aim for < 1 Mbps)"
echo -e "${GREEN}Quality:${NC} Maintain PSNR > 95%"
echo ""
echo -e "${BLUE}Video Characteristics:${NC}"
echo "• Resolution: 1920x1080 (1080p)"
echo "• Frame Rate: 30 FPS"
echo "• Duration: 10 seconds"
echo "• Content: Geometric test patterns with subtle noise and effects"
echo "• Challenge: High-frequency details, contrast, and motion"
echo ""
echo -e "${BLUE}Timeout Protection:${NC}"
echo "• SOURCE HD RAW: 60s timeout"
echo "• HEVC HD: 120s timeout"
echo "• All operations fail fast if they take too long"
echo ""
echo -e "${GREEN}Ready for AI codec testing! 🎬🚀${NC}"
