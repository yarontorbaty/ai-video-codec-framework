"""
Data loader for iPhone LiDAR-captured videos
Loads .mov files with RGB + depth tracks from LumaFlow app
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import av
from pathlib import Path
from typing import Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class iPhoneLiDARDataset(Dataset):
    """
    Dataset for iPhone-captured videos with LiDAR depth
    Expects .mov files with:
      - Track 0: HEVC-encoded RGB video
      - Track 1: HEVC-encoded depth data (float32)
    """
    
    def __init__(
        self,
        video_dir: str,
        frame_size: Tuple[int, int] = (512, 512),
        max_frames_per_video: Optional[int] = None,
        skip_frames: int = 1
    ):
        """
        Args:
            video_dir: Directory containing .mov files from iPhone
            frame_size: Target (width, height) for frames
            max_frames_per_video: Limit frames per video (for testing)
            skip_frames: Skip every N frames (for faster training)
        """
        self.video_dir = Path(video_dir)
        self.frame_size = frame_size
        self.max_frames_per_video = max_frames_per_video
        self.skip_frames = skip_frames
        
        # Find all .mov files
        self.video_files = sorted(list(self.video_dir.glob("*.mov")))
        
        if len(self.video_files) == 0:
            raise ValueError(f"No .mov files found in {video_dir}")
        
        logger.info(f"Found {len(self.video_files)} video files")
        
        # Build frame index: list of (video_idx, frame_idx) tuples
        self.frame_index = []
        self._build_frame_index()
        
        logger.info(f"Total frames available: {len(self.frame_index)}")
    
    def _build_frame_index(self):
        """Build index of all frames across all videos"""
        for video_idx, video_path in enumerate(self.video_files):
            try:
                # Open video to count frames
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # Limit frames if specified
                if self.max_frames_per_video:
                    total_frames = min(total_frames, self.max_frames_per_video)
                
                # Add frames with skip
                for frame_idx in range(0, total_frames, self.skip_frames + 1):
                    self.frame_index.append((video_idx, frame_idx))
                
                logger.info(f"  {video_path.name}: {total_frames} frames")
                
            except Exception as e:
                logger.warning(f"  Skipping {video_path.name}: {e}")
    
    def __len__(self) -> int:
        return len(self.frame_index)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            Dictionary with:
                - rgb: (3, H, W) tensor in [-1, 1]
                - depth: (1, H, W) tensor in [0, 1]
                - video_id: int
                - frame_id: int
        """
        video_idx, frame_idx = self.frame_index[idx]
        video_path = self.video_files[video_idx]
        
        # Load RGB and depth
        rgb_frame, depth_frame = self._load_frame(video_path, frame_idx)
        
        # Convert to tensors
        rgb_tensor = self._preprocess_rgb(rgb_frame)
        depth_tensor = self._preprocess_depth(depth_frame)
        
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'video_id': video_idx,
            'frame_id': frame_idx
        }
    
    def _load_frame(
        self,
        video_path: Path,
        frame_idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load RGB and depth frame from .mov file
        
        Returns:
            rgb_frame: (H, W, 3) numpy array
            depth_frame: (H, W) numpy array or None
        """
        try:
            # Use PyAV for multi-track video
            container = av.open(str(video_path))
            
            # Find video streams
            video_stream = container.streams.video[0]
            depth_stream = container.streams.video[1] if len(container.streams.video) > 1 else None
            
            # Seek to frame
            # Note: PyAV seeking can be imprecise, may need to iterate
            rgb_frame = None
            depth_frame = None
            
            for packet in container.demux(video_stream):
                for frame in packet.decode():
                    if frame.index == frame_idx:
                        rgb_frame = frame.to_ndarray(format='rgb24')
                        break
                if rgb_frame is not None:
                    break
            
            # Load depth if available
            if depth_stream:
                for packet in container.demux(depth_stream):
                    for frame in packet.decode():
                        if frame.index == frame_idx:
                            # Depth is stored as grayscale, convert to float
                            depth_array = frame.to_ndarray(format='gray')
                            depth_frame = depth_array.astype(np.float32)
                            break
                    if depth_frame is not None:
                        break
            
            container.close()
            
            # Fallback to OpenCV if PyAV fails
            if rgb_frame is None:
                rgb_frame = self._load_frame_opencv(video_path, frame_idx)
            
            return rgb_frame, depth_frame
            
        except Exception as e:
            logger.warning(f"Error loading frame {frame_idx} from {video_path.name}: {e}")
            # Return black frame as fallback
            return np.zeros((480, 640, 3), dtype=np.uint8), None
    
    def _load_frame_opencv(self, video_path: Path, frame_idx: int) -> np.ndarray:
        """Fallback: load RGB frame using OpenCV"""
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def _preprocess_rgb(self, rgb_frame: np.ndarray) -> torch.Tensor:
        """Convert RGB frame to normalized tensor"""
        # Resize
        rgb_resized = cv2.resize(rgb_frame, self.frame_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to float [0, 1]
        rgb_float = rgb_resized.astype(np.float32) / 255.0
        
        # To tensor (H, W, C) -> (C, H, W)
        rgb_tensor = torch.from_numpy(rgb_float).permute(2, 0, 1)
        
        # Normalize to [-1, 1]
        rgb_tensor = rgb_tensor * 2.0 - 1.0
        
        return rgb_tensor
    
    def _preprocess_depth(self, depth_frame: Optional[np.ndarray]) -> torch.Tensor:
        """Convert depth frame to normalized tensor"""
        if depth_frame is None:
            # Return zeros if no depth available
            return torch.zeros(1, self.frame_size[1], self.frame_size[0])
        
        # Resize
        depth_resized = cv2.resize(depth_frame, self.frame_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        depth_min, depth_max = depth_resized.min(), depth_resized.max()
        if depth_max > depth_min:
            depth_normalized = (depth_resized - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_resized)
        
        # To tensor (H, W) -> (1, H, W)
        depth_tensor = torch.from_numpy(depth_normalized).unsqueeze(0)
        
        return depth_tensor


def create_iphone_dataloader(
    video_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader for iPhone LiDAR videos
    
    Args:
        video_dir: Directory with .mov files from iPhone
        batch_size: Batch size for training
        num_workers: Number of worker processes
        shuffle: Shuffle data
        **kwargs: Additional args for iPhoneLiDARDataset
        
    Returns:
        DataLoader instance
    """
    dataset = iPhoneLiDARDataset(video_dir, **kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"Created DataLoader:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Num workers: {num_workers}")
    logger.info(f"  Total batches: {len(dataloader)}")
    
    return dataloader


# Test/demo code
if __name__ == "__main__":
    print("🎥 iPhone LiDAR Dataset Test")
    print("=" * 50)
    
    # Create dummy test data
    import os
    test_dir = "/tmp/test_iphone_videos"
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"\n📁 Test directory: {test_dir}")
    print("Note: Place .mov files from iPhone here to test")
    print("\nExpected file structure:")
    print("  /tmp/test_iphone_videos/")
    print("  ├── lumaflow_1.mov  (RGB + depth)")
    print("  ├── lumaflow_2.mov")
    print("  └── ...")
    
    # Try to load if files exist
    if list(Path(test_dir).glob("*.mov")):
        print("\n✅ Found videos, testing dataloader...")
        
        try:
            # Create dataset
            dataset = iPhoneLiDARDataset(
                test_dir,
                max_frames_per_video=10,  # Limit for testing
                skip_frames=0
            )
            
            # Test loading a frame
            sample = dataset[0]
            print(f"\n📊 Sample frame:")
            print(f"  RGB shape: {sample['rgb'].shape}")
            print(f"  RGB range: [{sample['rgb'].min():.2f}, {sample['rgb'].max():.2f}]")
            print(f"  Depth shape: {sample['depth'].shape}")
            print(f"  Depth range: [{sample['depth'].min():.2f}, {sample['depth'].max():.2f}]")
            
            # Create dataloader
            dataloader = create_iphone_dataloader(
                test_dir,
                batch_size=2,
                max_frames_per_video=10
            )
            
            # Test batch
            batch = next(iter(dataloader))
            print(f"\n📦 Sample batch:")
            print(f"  RGB batch shape: {batch['rgb'].shape}")
            print(f"  Depth batch shape: {batch['depth'].shape}")
            
            print("\n✅ Dataset test passed!")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠️  No .mov files found in {test_dir}")
        print("Capture some videos with the LumaFlow iPhone app first!")

