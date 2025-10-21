#!/bin/bash
set -e

echo "🧹 Creating clean PVC v2.0 branch..."

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo "Temp: $TEMP_DIR"

# Copy ONLY PVC code files (no videos)
cp -r 1-PVC-v2.0/pvc_v2 "$TEMP_DIR/"
cp -r 1-PVC-v2.0/pvc_research "$TEMP_DIR/"

# Remove video files from temp (they're large)
find "$TEMP_DIR/pvc_research" -name "*.mp4" -delete 2>/dev/null || true

# Copy documentation
cp 1-PVC-v2.0/README.md "$TEMP_DIR/"
cp 1-PVC-v2.0/MODEL_DOWNLOAD.md "$TEMP_DIR/"
cp 1-PVC-v2.0/SOTA_FULL_TRAINING_RESULTS.md "$TEMP_DIR/"
cp 1-PVC-v2.0/PVC_*.md "$TEMP_DIR/" 2>/dev/null || true

# Create gitignore
cat > "$TEMP_DIR/.gitignore" << 'GITIGNORE'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store

# Training outputs & large files
*.pth
*.pt
*.mp4
*.avi
/tmp/

# Keep comparison images (small)
!pvc_v2/tests/*.png
!pvc_v2/tests/*.jpg
GITIGNORE

echo "✅ PVC files prepared"

# Create orphan branch
git checkout --orphan pvc-v2.0-clean
git rm -rf . 2>/dev/null || true

# Copy files to root
cp -r "$TEMP_DIR"/* "$TEMP_DIR"/.gitignore .

# Stage
git add .

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "✅ Ready! Files staged:"
git status --short | head -20
echo "..."
echo ""
echo "Next: git commit && git push"
