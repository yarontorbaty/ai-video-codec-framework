#!/usr/bin/env python3
"""
Quick test video generator - creates smaller, faster test videos
"""

import cv2
import numpy as np
import os
import math
from pathlib import Path

def create_test_video(output_path, width=1920, height=1080, duration=5, fps=30):
    """Create a simple but challenging test video"""
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    print(f"Generating {total_frames} frames at {width}x{height} @ {fps}fps...")
    
    for frame_num in range(total_frames):
        # Create a frame with challenging patterns
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                # Complex wave patterns that are hard to compress
                wave1 = math.sin(x * 0.1 + frame_num * 0.1) * 50
                wave2 = math.cos(y * 0.15 + frame_num * 0.05) * 30
                wave3 = math.sin((x + y) * 0.05 + frame_num * 0.2) * 25
                
                # Add some noise for high-frequency content
                noise = np.random.normal(0, 5)
                
                intensity = int(128 + wave1 + wave2 + wave3 + noise)
                intensity = max(0, min(255, intensity))
                
                # Color variation
                r = intensity
                g = max(0, min(255, intensity + int(20 * math.sin(frame_num * 0.1))))
                b = max(0, min(255, intensity + int(15 * math.cos(frame_num * 0.15))))
                
                frame[y, x] = [r, g, b]
        
        out.write(frame)
        
        if frame_num % (total_frames // 5) == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    out.release()
    print(f"Video saved: {output_path}")

def main():
    # Create test data directory
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    print("🎬 Quick Test Video Generator")
    print("=" * 40)
    
    # Generate source video (1080p, 5 seconds)
    source_path = test_dir / "source_4k60_10s.mp4"
    print("Creating source video...")
    create_test_video(str(source_path), 1920, 1080, 5, 30)
    
    # Create HEVC reference using FFmpeg if available
    hevc_path = test_dir / "hevc_4k60_10s.mp4"
    if os.system("which ffmpeg > /dev/null 2>&1") == 0:
        print("Creating HEVC reference...")
        os.system(f"ffmpeg -i {source_path} -c:v libx265 -crf 23 -preset fast -y {hevc_path}")
        print(f"HEVC reference saved: {hevc_path}")
    else:
        print("FFmpeg not found, skipping HEVC reference")
    
    print("\n✅ Test videos created!")
    print(f"📁 Files: {test_dir}")
    print("🎯 Ready for upload to S3!")

if __name__ == "__main__":
    main()
