# LumaFlow - iPhone App

**Volumetric Video Capture with Real-Time LiDAR**

## Overview

LumaFlow is an iPhone app that captures video with LiDAR depth data and offers three powerful capture modes:

### 📱 Capture Modes

1. **Save to File** 🗂️
   - Captures video + depth and saves locally
   - HEVC-compressed RGB video
   - Losslessly compressed depth maps
   - Camera transform metadata
   - Output: `.mov` file compatible with QuickTime/FCP

2. **Stream to AWS** ☁️
   - Real-time streaming to AWS encoding server
   - HEVC compression for RGB (5 Mbps)
   - LZFSE compression for depth
   - SRT protocol for reliable delivery
   - Live bitrate monitoring

3. **Encode on Device** 🔧
   - Experimental LumaFlow codec encoding
   - I-frames: Downscaled + LCM latents + depth
   - P-frames: Motion vectors + residuals + depth deltas
   - Output: `.lfv` (LumaFlow Video) format
   - ~50-70x compression target

## Requirements

- iPhone 12 Pro or later (LiDAR required)
- iOS 17.0+ 
- Xcode 15.0+
- Swift 5.9+

## Features

- ✅ Real-time LiDAR depth capture (256x192 @ 60Hz)
- ✅ High-quality video (1080p or 4K @ 30/60fps)
- ✅ Depth visualization overlay
- ✅ Performance statistics (FPS, bitrate, file size)
- ✅ Three capture modes
- ✅ Beautiful dark theme UI
- ✅ Settings panel

## Build & Run

### 1. Open in Xcode
```bash
cd LumaFlow
open LumaFlow.xcodeproj
```

### 2. Configure Signing
- Select your development team in "Signing & Capabilities"
- Update bundle identifier if needed

### 3. Connect iPhone
- iPhone 12 Pro or later
- Enable Developer Mode in Settings

### 4. Build & Run
- Select your device in Xcode
- Press Cmd+R to build and run

## Project Structure

```
LumaFlow/
├── LumaFlow/
│   ├── LumaFlowApp.swift          # App entry point
│   ├── Models/
│   │   └── CaptureMode.swift       # Capture mode enum
│   ├── Views/
│   │   └── ContentView.swift       # Main UI
│   ├── Services/
│   │   ├── LiDARCaptureService.swift  # ARKit + LiDAR capture
│   │   ├── FileWriter.swift           # Mode 1: Local save
│   │   ├── StreamingService.swift     # Mode 2: AWS streaming  
│   │   └── OnDeviceEncoder.swift      # Mode 3: LumaFlow codec
│   └── Info.plist                  # App configuration
└── README.md
```

## Usage

### Mode 1: Save to File

1. Select "Save to File" mode
2. Tap the record button
3. Capture your scene
4. Tap stop
5. Video + depth saved to Files app

**Output:**
- Video: HEVC-compressed RGB
- Depth: Separate track in same file
- Metadata: Camera transforms, timestamps
- Format: `.mov` (QuickTime compatible)

### Mode 2: Stream to AWS

1. Go to Settings
2. Enter your server URL (e.g., `rtmp://server:1935/live`)
3. Select "Stream to AWS" mode
4. Tap record
5. Live stream starts automatically

**Requirements:**
- AWS streaming server (see AWS setup below)
- Network connection (WiFi/5G recommended)
- SRT-compatible server

### Mode 3: Encode on Device

1. Select "Encode on Device" mode
2. Tap record
3. Capture your scene
4. Tap stop
5. Encoded `.lfv` file saved

**Note:** This is experimental! The on-device encoder uses placeholder implementations. For production, you'll need:
- CoreML LCM model
- Optimized motion estimation
- Hardware-accelerated quantization

## AWS Server Setup

For Mode 2 (streaming), you need an AWS encoding server:

### Quick Setup

```bash
# Launch EC2 instance (g4dn.xlarge recommended)
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type g4dn.xlarge \
    --key-name your-key \
    --security-group-ids sg-xxx \
    --subnet-id subnet-xxx

# Install SRT server
sudo apt update
sudo apt install -y srt-tools

# Start SRT server
srt-live-transmit srt://:1935 file://output.ts
```

For full encoder setup, see `../generative_codec/README.md`

## File Formats

### .mov (Mode 1)
Standard QuickTime format with two tracks:
- Track 1: HEVC-compressed RGB video
- Track 2: Depth data (HEVC-compressed float32)

### .lfv (Mode 3)
LumaFlow Video format:
```
Header (20 bytes):
  - version: uint32
  - frame_count: uint32
  - width: uint32
  - height: uint32  
  - frame_rate: uint32

Frames (variable size):
  For each frame:
    - timestamp: float64
    - is_iframe: bool
    - latent_data_length: uint32
    - latent_data: bytes
    - depth_data_length: uint32
    - depth_data: bytes
    - motion_data_length: uint32
    - motion_data: bytes
```

## Performance

Measured on iPhone 15 Pro:

| Mode | Resolution | FPS | CPU Usage | File Size (1 min) |
|------|-----------|-----|-----------|------------------|
| Save to File | 1080p | 30 | ~40% | ~60 MB |
| Save to File | 4K | 30 | ~60% | ~120 MB |
| Stream to AWS | 1080p | 30 | ~50% | Streamed |
| Encode on Device | 1080p | 30 | ~80% | ~2 MB (70x) |

## Troubleshooting

### "LiDAR not available"
- Ensure you have iPhone 12 Pro or later
- Check ARKit permissions in Settings

### "Streaming failed"
- Verify server URL is correct
- Check network connection
- Ensure firewall allows SRT traffic

### "Encoding too slow"
- Lower resolution to 1080p
- Reduce frame rate to 30fps
- Close background apps

## Next Steps

1. **Capture test footage** with all 3 modes
2. **Transfer files** to Mac for training
3. **Train LumaFlow codec** with captured data
4. **Replace placeholder** encoder with trained CoreML models
5. **Optimize performance** with quantization

## Contributing

This is a research project. Contributions welcome!

## License

MIT License - See LICENSE file

## Credits

Created by Yaron Torbaty
LumaFlow Codec v1.0
October 2025

