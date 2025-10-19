#!/bin/bash
# Cache PVC reference videos on worker instance

set -e

CACHE_DIR="/home/ec2-user/pvc/cache"
S3_BUCKET="ai-codec-v3-artifacts-580473065386"
S3_PREFIX="pvc/reference"

echo "🗄️  Setting up PVC reference video cache..."
echo ""

# Create cache directory
mkdir -p "$CACHE_DIR"
cd "$CACHE_DIR"

# List of files to cache
FILES=(
    "source_anime_01.mp4"
    "source_anime_02.mp4"
    "source_anime_03.mp4"
    "av1_anime_01.mp4"
    "av1_anime_02.mp4"
    "av1_anime_03.mp4"
)

echo "📁 Cache directory: $CACHE_DIR"
echo ""

# Download each file (or verify if already cached)
for file in "${FILES[@]}"; do
    s3_path="s3://${S3_BUCKET}/${S3_PREFIX}/${file}"
    local_path="${CACHE_DIR}/${file}"
    
    if [ -f "$local_path" ]; then
        echo "✓ $file (already cached)"
        
        # Verify size matches S3
        local_size=$(stat -f%z "$local_path" 2>/dev/null || stat -c%s "$local_path" 2>/dev/null)
        s3_size=$(aws s3api head-object --bucket "$S3_BUCKET" --key "${S3_PREFIX}/${file}" --query ContentLength --output text 2>/dev/null)
        
        if [ "$local_size" = "$s3_size" ]; then
            echo "  Size match: $(numfmt --to=iec-i --suffix=B $local_size 2>/dev/null || echo ${local_size} bytes)"
        else
            echo "  ⚠️  Size mismatch (local: $local_size, S3: $s3_size)"
            echo "  Re-downloading..."
            aws s3 cp "$s3_path" "$local_path" --region us-east-1 --quiet
            echo "  ✅ Re-downloaded"
        fi
    else
        echo "⬇️  Downloading $file..."
        aws s3 cp "$s3_path" "$local_path" --region us-east-1 --quiet
        
        if [ -f "$local_path" ]; then
            size=$(stat -f%z "$local_path" 2>/dev/null || stat -c%s "$local_path" 2>/dev/null)
            size_human=$(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "${size} bytes")
            echo "  ✅ Downloaded ($size_human)"
        else
            echo "  ❌ Failed to download"
            exit 1
        fi
    fi
    echo ""
done

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Cache ready!"
echo ""
echo "📊 Cached files:"
ls -lh "$CACHE_DIR" | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}'
echo ""

total_size=$(du -sh "$CACHE_DIR" | awk '{print $1}')
echo "Total cache size: $total_size"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

