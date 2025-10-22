# LumaFlow Depth Analysis

This tool extracts and visualizes depth data from LumaFlow recordings.

## Prerequisites

```bash
# Install Python dependencies
pip install opencv-python numpy matplotlib

# Install ffmpeg (for extracting depth track)
brew install ffmpeg
```

## Usage

### 1. Transfer Video from iPhone

**Via AirDrop:**
- Open Photos app on iPhone
- Find your LumaFlow video
- Share → AirDrop to Mac
- Save to this directory

### 2. Basic Analysis

```bash
# View RGB frames (no depth extraction needed)
python analyze_depth.py path/to/lumaflow_*.mov

# Extract depth track and analyze
python analyze_depth.py path/to/lumaflow_*.mov --extract-depth

# Visualize specific frame
python analyze_depth.py path/to/lumaflow_*.mov --extract-depth --frame 30

# Show depth statistics
python analyze_depth.py path/to/lumaflow_*.mov --extract-depth --stats
```

### 3. Output Files

The script creates:
- `depth_track.mov` - Extracted depth video track
- `frame_XXXX_analysis.png` - Side-by-side visualization (RGB + Depth + Heatmap)
- `depth_histogram.png` - Depth value distribution

## Video Structure

LumaFlow recordings contain 2 video tracks:

**Track 0 (RGB):**
- Resolution: 1920×1080
- Codec: HEVC
- Bitrate: 10 Mbps

**Track 1 (Depth):**
- Resolution: 256×192
- Codec: HEVC (grayscale)
- Bitrate: 2 Mbps
- Contains: Float32 depth values from LiDAR

## Advanced Usage

### Extract Depth Track Only
```bash
ffmpeg -i lumaflow_*.mov -map 0:1 -c copy depth_only.mov
```

### View Track Info
```bash
ffprobe -v error -show_streams lumaflow_*.mov
```

### Convert Depth to Images
```python
import cv2
cap = cv2.VideoCapture('depth_track.mov')
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    depth = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'depth_{frame_idx:04d}.png', depth)
    frame_idx += 1
```

## Interpreting Depth Values

The depth track is encoded as grayscale HEVC:
- **Darker pixels** = Closer objects
- **Lighter pixels** = Farther objects
- Values range from 0-255 in the encoded format
- Original LiDAR depth is in meters (0-5m typically)

## Tips

1. **Frame Selection**: Use `--frame N` to analyze specific moments
2. **Statistics**: Use `--stats` to see depth value distribution
3. **Batch Processing**: Create a loop to process multiple videos
4. **Performance**: Depth track is smaller (256×192) for efficiency

## Troubleshooting

**"Could not open video file"**
- Check file path
- Ensure video was saved successfully from app

**"ffmpeg not found"**
- Install: `brew install ffmpeg`

**"No depth track found"**
- Use `--extract-depth` flag
- Check if video has 2 tracks: `ffprobe file.mov`

## Next Steps

1. Integrate depth data into your codec
2. Analyze compression efficiency
3. Compare depth quality across frames
4. Build 3D reconstruction pipeline

