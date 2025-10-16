#!/usr/bin/env python3
"""
Video Utilities for AI Video Codec Framework
Handles video processing, frame extraction, and format conversion.
"""

import cv2
import numpy as np
import logging
import os
import subprocess
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Video processing utilities."""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information using OpenCV."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'codec': self._fourcc_to_string(int(cap.get(cv2.CAP_PROP_FOURCC)))
        }
        
        cap.release()
        return info
    
    def _fourcc_to_string(self, fourcc: int) -> str:
        """Convert fourcc code to string."""
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    def extract_frames(self, video_path: str, output_dir: str, 
                      max_frames: Optional[int] = None) -> List[str]:
        """Extract frames from video."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        frame_paths = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"Extracted {frame_count} frames")
        
        cap.release()
        logger.info(f"Extracted {len(frame_paths)} frames to {output_dir}")
        return frame_paths
    
    def create_video_from_frames(self, frame_paths: List[str], output_path: str, 
                                fps: float = 30.0, codec: str = 'mp4v') -> bool:
        """Create video from frame images."""
        if not frame_paths:
            logger.error("No frame paths provided")
            return False
        
        # Get frame dimensions from first frame
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            logger.error(f"Could not read first frame: {frame_paths[0]}")
            return False
        
        height, width = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"Could not create video writer for {output_path}")
            return False
        
        # Write frames
        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.warning(f"Could not read frame {i}: {frame_path}")
                continue
            
            out.write(frame)
            
            if i % 100 == 0:
                logger.info(f"Written {i}/{len(frame_paths)} frames")
        
        out.release()
        logger.info(f"Created video: {output_path}")
        return True
    
    def resize_video(self, input_path: str, output_path: str, 
                    new_size: Tuple[int, int]) -> bool:
        """Resize video to new dimensions."""
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open input video: {input_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        # Create output video writer
        out = cv2.VideoWriter(output_path, fourcc, fps, new_size)
        
        if not out.isOpened():
            logger.error(f"Could not create output video: {output_path}")
            cap.release()
            return False
        
        # Process frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            resized_frame = cv2.resize(frame, new_size)
            out.write(resized_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        cap.release()
        out.release()
        
        logger.info(f"Resized video saved to: {output_path}")
        return True
    
    def convert_format(self, input_path: str, output_path: str, 
                      codec: str = 'mp4v', quality: int = 23) -> bool:
        """Convert video format using OpenCV."""
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open input video: {input_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"Could not create output video: {output_path}")
            cap.release()
            return False
        
        # Process frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"Converted {frame_count} frames")
        
        cap.release()
        out.release()
        
        logger.info(f"Video converted to: {output_path}")
        return True
    
    def get_frame_at_time(self, video_path: str, time_seconds: float) -> Optional[np.ndarray]:
        """Get frame at specific time."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
        
        # Calculate frame number
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(time_seconds * fps)
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read frame
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        else:
            logger.error(f"Could not read frame at time {time_seconds}s")
            return None
    
    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """Extract audio from video using ffmpeg."""
        try:
            cmd = [
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'copy', 
                '-y', audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Audio extracted to: {audio_path}")
                return True
            else:
                logger.error(f"Audio extraction failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.error("ffmpeg not found. Please install ffmpeg.")
            return False
    
    def add_audio_to_video(self, video_path: str, audio_path: str, 
                          output_path: str) -> bool:
        """Add audio to video using ffmpeg."""
        try:
            cmd = [
                'ffmpeg', '-i', video_path, '-i', audio_path,
                '-c:v', 'copy', '-c:a', 'aac', '-y', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Video with audio saved to: {output_path}")
                return True
            else:
                logger.error(f"Audio addition failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.error("ffmpeg not found. Please install ffmpeg.")
            return False
    
    def validate_video(self, video_path: str) -> bool:
        """Validate video file integrity."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return False
        
        # Try to read first few frames
        for i in range(5):
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Could not read frame {i}")
                cap.release()
                return False
        
        cap.release()
        logger.info(f"Video validation successful: {video_path}")
        return True
    
    def get_video_thumbnail(self, video_path: str, output_path: str, 
                          time_seconds: float = 5.0) -> bool:
        """Extract thumbnail from video."""
        frame = self.get_frame_at_time(video_path, time_seconds)
        
        if frame is None:
            return False
        
        success = cv2.imwrite(output_path, frame)
        
        if success:
            logger.info(f"Thumbnail saved to: {output_path}")
        else:
            logger.error(f"Could not save thumbnail: {output_path}")
        
        return success
    
    def calculate_video_hash(self, video_path: str) -> str:
        """Calculate hash of video file for integrity checking."""
        import hashlib
        
        hash_md5 = hashlib.md5()
        
        with open(video_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def get_video_statistics(self, video_path: str) -> Dict[str, Any]:
        """Get comprehensive video statistics."""
        info = self.get_video_info(video_path)
        
        # File statistics
        file_size = os.path.getsize(video_path)
        
        # Calculate bitrate
        duration = info['duration']
        bitrate_bps = (file_size * 8) / duration if duration > 0 else 0
        bitrate_mbps = bitrate_bps / 1_000_000
        
        # Calculate compression ratio (rough estimate)
        uncompressed_size = info['width'] * info['height'] * 3 * info['frame_count']
        compression_ratio = file_size / uncompressed_size if uncompressed_size > 0 else 0
        
        stats = {
            **info,
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'bitrate_bps': bitrate_bps,
            'bitrate_mbps': bitrate_mbps,
            'compression_ratio': compression_ratio,
            'file_hash': self.calculate_video_hash(video_path)
        }
        
        return stats
