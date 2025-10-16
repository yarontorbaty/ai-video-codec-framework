#!/usr/bin/env python3
"""
AI Video Codec Framework - Test Video Generator
Generates challenging test videos that stress traditional encoders
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import math

def create_complex_pattern(width, height, frame_num, pattern_type="mixed"):
    """Create complex patterns that are hard to encode efficiently"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    if pattern_type == "noise":
        # High-frequency noise - very hard to compress
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
    elif pattern_type == "texture":
        # Fine textures and patterns
        for y in range(height):
            for x in range(width):
                # Complex wave patterns
                wave1 = math.sin(x * 0.1 + frame_num * 0.1) * 50
                wave2 = math.cos(y * 0.15 + frame_num * 0.05) * 30
                wave3 = math.sin((x + y) * 0.05 + frame_num * 0.2) * 25
                
                intensity = int(128 + wave1 + wave2 + wave3)
                intensity = max(0, min(255, intensity))
                frame[y, x] = [intensity, intensity//2, 255-intensity]
                
    elif pattern_type == "motion":
        # Fast motion with fine details
        center_x = width // 2 + int(50 * math.sin(frame_num * 0.3))
        center_y = height // 2 + int(30 * math.cos(frame_num * 0.2))
        
        for y in range(height):
            for x in range(width):
                dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                angle = math.atan2(y - center_y, x - center_x)
                
                # Spiral pattern
                spiral = math.sin(dist * 0.1 + angle * 3 + frame_num * 0.5) * 100
                intensity = int(128 + spiral)
                intensity = max(0, min(255, intensity))
                frame[y, x] = [intensity, 255-intensity, intensity//2]
                
    elif pattern_type == "mixed":
        # Combination of all challenging elements
        # High-frequency details
        for y in range(height):
            for x in range(width):
                # Multiple overlapping patterns
                pattern1 = math.sin(x * 0.2) * math.cos(y * 0.2) * 50
                pattern2 = math.sin((x + y) * 0.1 + frame_num * 0.1) * 30
                pattern3 = math.sin(x * 0.05) * math.sin(y * 0.05) * 20
                
                # Add some noise
                noise = np.random.normal(0, 10)
                
                # Motion blur effect
                motion_offset = int(frame_num * 2)
                if x + motion_offset < width:
                    pattern4 = math.sin((x + motion_offset) * 0.15) * 25
                else:
                    pattern4 = 0
                
                intensity = int(128 + pattern1 + pattern2 + pattern3 + pattern4 + noise)
                intensity = max(0, min(255, intensity))
                
                # Color variation
                r = intensity
                g = max(0, min(255, intensity + int(20 * math.sin(frame_num * 0.1))))
                b = max(0, min(255, intensity + int(15 * math.cos(frame_num * 0.15))))
                
                frame[y, x] = [r, g, b]
    
    return frame

def create_natural_scene(width, height, frame_num):
    """Create natural-looking scenes that are still challenging to encode"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Simulate natural textures
    for y in range(height):
        for x in range(width):
            # Sky gradient
            sky_intensity = int(200 - (y / height) * 100)
            
            # Cloud patterns
            cloud_noise = np.random.normal(0, 15)
            cloud_pattern = math.sin(x * 0.01 + y * 0.005 + frame_num * 0.02) * 20
            
            # Ground texture
            if y > height * 0.7:
                ground_intensity = int(100 + (y - height * 0.7) / (height * 0.3) * 50)
                texture = math.sin(x * 0.05 + y * 0.03) * 10
                intensity = max(0, min(255, ground_intensity + texture + cloud_noise))
                frame[y, x] = [intensity//2, intensity, intensity//3]
            else:
                # Sky with clouds
                intensity = max(0, min(255, sky_intensity + cloud_pattern + cloud_noise))
                frame[y, x] = [intensity, intensity, 255]
    
    return frame

def create_test_video(output_path, width=3840, height=2160, duration=10, fps=60, pattern_type="mixed"):
    """Create a test video with specified parameters"""
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    print(f"Generating {total_frames} frames at {width}x{height} @ {fps}fps...")
    
    for frame_num in range(total_frames):
        if pattern_type == "natural":
            frame = create_natural_scene(width, height, frame_num)
        else:
            frame = create_complex_pattern(width, height, frame_num, pattern_type)
        
        out.write(frame)
        
        if frame_num % (total_frames // 10) == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    out.release()
    print(f"Video saved: {output_path}")

def create_hevc_reference(input_path, output_path, crf=23):
    """Create HEVC reference using FFmpeg"""
    import subprocess
    
    cmd = [
        'ffmpeg', '-i', input_path,
        '-c:v', 'libx265',
        '-crf', str(crf),
        '-preset', 'medium',
        '-pix_fmt', 'yuv420p',
        '-y',  # Overwrite output
        output_path
    ]
    
    print(f"Creating HEVC reference with CRF {crf}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"HEVC reference saved: {output_path}")
        return True
    else:
        print(f"Error creating HEVC reference: {result.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate test videos for AI codec testing')
    parser.add_argument('--output-dir', default='test_data', help='Output directory')
    parser.add_argument('--width', type=int, default=3840, help='Video width')
    parser.add_argument('--height', type=int, default=2160, help='Video height')
    parser.add_argument('--duration', type=int, default=10, help='Duration in seconds')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second')
    parser.add_argument('--patterns', nargs='+', 
                       choices=['noise', 'texture', 'motion', 'mixed', 'natural'],
                       default=['mixed', 'natural'],
                       help='Pattern types to generate')
    parser.add_argument('--create-hevc', action='store_true', 
                       help='Create HEVC reference videos')
    parser.add_argument('--hevc-crf', type=int, default=23,
                       help='HEVC CRF value (lower = better quality)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🎬 AI Video Codec Test Video Generator")
    print("=" * 50)
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Duration: {args.duration}s @ {args.fps}fps")
    print(f"Patterns: {', '.join(args.patterns)}")
    print()
    
    for pattern in args.patterns:
        print(f"Generating {pattern} pattern video...")
        
        # Source video (uncompressed)
        source_path = output_dir / f"source_4k60_10s_{pattern}.mp4"
        create_test_video(
            str(source_path), 
            args.width, 
            args.height, 
            args.duration, 
            args.fps, 
            pattern
        )
        
        # HEVC reference
        if args.create_hevc:
            hevc_path = output_dir / f"hevc_4k60_10s_{pattern}.mp4"
            create_hevc_reference(str(source_path), str(hevc_path), args.hevc_crf)
    
    print("\n✅ Test video generation complete!")
    print(f"📁 Files saved to: {output_dir}")
    print("\n📊 Video Characteristics:")
    print("• High-frequency details - stress DCT-based encoders")
    print("• Complex motion patterns - challenge motion estimation")
    print("• Fine textures - test spatial prediction")
    print("• Natural scenes - realistic compression scenarios")
    print("\n🎯 Perfect for testing AI codec performance!")

if __name__ == "__main__":
    main()
