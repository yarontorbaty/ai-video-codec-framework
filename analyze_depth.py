#!/usr/bin/env python3
"""
LumaFlow Depth Analysis Tool
Extracts and visualizes depth data from LumaFlow recordings
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def extract_tracks(video_path):
    """Extract RGB and Depth tracks from LumaFlow video."""
    print(f"📹 Opening: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError("❌ Could not open video file")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ Video info:")
    print(f"   - FPS: {fps}")
    print(f"   - Frames: {frame_count}")
    print(f"   - Resolution: {width}x{height}")
    
    rgb_frames = []
    depth_frames = []
    
    # Extract frames
    print("📊 Extracting frames...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # First track is RGB (1920x1080)
        # Note: OpenCV only reads one track by default
        # We'll need to use ffmpeg to extract both tracks separately
        rgb_frames.append(frame)
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            print(f"   Processed {frame_idx}/{frame_count} frames...")
    
    cap.release()
    
    print(f"✅ Extracted {len(rgb_frames)} RGB frames")
    return np.array(rgb_frames), fps


def extract_depth_track_ffmpeg(video_path, output_path="depth_track.mov"):
    """Extract depth track using ffmpeg."""
    import subprocess
    
    print(f"🎬 Extracting depth track with ffmpeg...")
    
    # First, check what tracks exist
    probe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v',
        '-show_entries', 'stream=index,codec_name,width,height',
        '-of', 'json',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        print("📹 Video tracks:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"⚠️  ffprobe failed: {e}")
        print(f"   Make sure ffmpeg is installed: brew install ffmpeg")
        return None
    
    # Extract second track (depth) - map 0:1 means stream 0, track 1
    extract_cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-map', '0:1',  # Select second video track (depth)
        '-c', 'copy',   # Copy codec without re-encoding
        '-y',           # Overwrite output
        str(output_path)
    ]
    
    try:
        subprocess.run(extract_cmd, check=True, capture_output=True)
        print(f"✅ Depth track extracted to: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to extract depth track: {e}")
        print(f"   stderr: {e.stderr.decode() if e.stderr else 'none'}")
        return None


def read_depth_frames(depth_video_path):
    """Read depth frames from extracted depth track."""
    print(f"📊 Reading depth data from: {depth_video_path}")
    
    cap = cv2.VideoCapture(str(depth_video_path))
    if not cap.isOpened():
        raise ValueError("❌ Could not open depth track")
    
    depth_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert grayscale HEVC to float depth values
        # The depth is encoded as grayscale, need to interpret it
        depth = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth_frames.append(depth)
    
    cap.release()
    print(f"✅ Extracted {len(depth_frames)} depth frames")
    return np.array(depth_frames)


def visualize_frame(rgb_frame, depth_frame, frame_idx, save_path=None):
    """Visualize RGB and depth side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # RGB
    axes[0].imshow(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'RGB Frame {frame_idx}')
    axes[0].axis('off')
    
    # Depth (grayscale)
    if depth_frame is not None:
        axes[1].imshow(depth_frame, cmap='gray')
        axes[1].set_title(f'Depth Frame {frame_idx}')
        axes[1].axis('off')
        
        # Depth (colored heatmap)
        depth_colored = axes[2].imshow(depth_frame, cmap='viridis')
        axes[2].set_title(f'Depth Heatmap {frame_idx}')
        axes[2].axis('off')
        plt.colorbar(depth_colored, ax=axes[2])
    else:
        axes[1].text(0.5, 0.5, 'No depth data', ha='center', va='center')
        axes[1].axis('off')
        axes[2].text(0.5, 0.5, 'Extract depth track first', ha='center', va='center')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization to: {save_path}")
    
    plt.show()


def analyze_depth_statistics(depth_frames):
    """Analyze depth data statistics."""
    print("\n📈 Depth Statistics:")
    print(f"   - Shape: {depth_frames.shape}")
    print(f"   - Min value: {depth_frames.min()}")
    print(f"   - Max value: {depth_frames.max()}")
    print(f"   - Mean value: {depth_frames.mean():.2f}")
    print(f"   - Std dev: {depth_frames.std():.2f}")
    
    # Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(depth_frames.flatten(), bins=100, alpha=0.7, edgecolor='black')
    plt.xlabel('Depth Value')
    plt.ylabel('Frequency')
    plt.title('Depth Value Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('depth_histogram.png', dpi=150)
    print("💾 Saved histogram to: depth_histogram.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze LumaFlow depth data')
    parser.add_argument('video', type=str, help='Path to LumaFlow .mov file')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to visualize')
    parser.add_argument('--extract-depth', action='store_true', help='Extract depth track with ffmpeg')
    parser.add_argument('--stats', action='store_true', help='Show depth statistics')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    print("🎬 LumaFlow Depth Analyzer")
    print("=" * 50)
    
    # Extract RGB frames
    rgb_frames, fps = extract_tracks(video_path)
    
    depth_frames = None
    depth_track_path = video_path.parent / "depth_track.mov"
    
    # Extract depth track if requested
    if args.extract_depth:
        depth_path = extract_depth_track_ffmpeg(video_path, depth_track_path)
        if depth_path:
            depth_frames = read_depth_frames(depth_path)
    elif depth_track_path.exists():
        # Use existing depth track
        print(f"📂 Using existing depth track: {depth_track_path}")
        depth_frames = read_depth_frames(depth_track_path)
    else:
        print("⚠️  No depth track found. Use --extract-depth to extract it.")
    
    # Visualize specific frame
    if args.frame < len(rgb_frames):
        depth_frame = depth_frames[args.frame] if depth_frames is not None and args.frame < len(depth_frames) else None
        output_path = f"frame_{args.frame:04d}_analysis.png"
        visualize_frame(rgb_frames[args.frame], depth_frame, args.frame, output_path)
    else:
        print(f"❌ Frame {args.frame} out of range (max: {len(rgb_frames)-1})")
    
    # Show statistics
    if args.stats and depth_frames is not None:
        analyze_depth_statistics(depth_frames)
    
    print("\n✅ Analysis complete!")
    print(f"📁 Output files in: {video_path.parent}")


if __name__ == '__main__':
    main()

