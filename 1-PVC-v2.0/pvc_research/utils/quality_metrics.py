"""
Quality Metrics - PVC Utils

Calculates video quality metrics (PSNR, SSIM, VMAF) for comparing
reconstructed video to original.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityMetrics:
    """
    Calculates quality metrics for video compression evaluation.
    """
    
    @staticmethod
    def calculate_psnr(original: np.ndarray,
                      reconstructed: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).
        
        Args:
            original: Original frame (H x W x 3)
            reconstructed: Reconstructed frame (H x W x 3)
            
        Returns:
            PSNR in dB (higher is better)
        """
        mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
        
        if mse == 0:
            return float('inf')  # Perfect reconstruction
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return float(psnr)
    
    @staticmethod
    def calculate_ssim(original: np.ndarray,
                      reconstructed: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM).
        
        Args:
            original: Original frame (H x W x 3)
            reconstructed: Reconstructed frame (H x W x 3)
            
        Returns:
            SSIM score in [0, 1] (higher is better)
        """
        # Convert to grayscale for SSIM
        if len(original.shape) == 3:
            gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            gray_recon = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
        else:
            gray_orig = original
            gray_recon = reconstructed
        
        # Calculate SSIM
        score, _ = ssim(gray_orig, gray_recon, full=True, data_range=255)
        
        return float(score)
    
    @staticmethod
    def calculate_frame_metrics(original: np.ndarray,
                               reconstructed: np.ndarray) -> Dict:
        """
        Calculate all metrics for a single frame.
        
        Args:
            original: Original frame
            reconstructed: Reconstructed frame
            
        Returns:
            Dictionary with PSNR and SSIM
        """
        # Ensure same size
        if original.shape != reconstructed.shape:
            reconstructed = cv2.resize(reconstructed, 
                                      (original.shape[1], original.shape[0]))
        
        psnr = QualityMetrics.calculate_psnr(original, reconstructed)
        ssim_score = QualityMetrics.calculate_ssim(original, reconstructed)
        
        return {
            'psnr_db': psnr,
            'ssim': ssim_score
        }
    
    @staticmethod
    def calculate_video_metrics(original_frames: List[np.ndarray],
                               reconstructed_frames: List[np.ndarray]) -> Dict:
        """
        Calculate metrics for entire video sequence.
        
        Args:
            original_frames: List of original frames
            reconstructed_frames: List of reconstructed frames
            
        Returns:
            Dictionary with average metrics
        """
        if len(original_frames) != len(reconstructed_frames):
            logger.warning(f"Frame count mismatch: {len(original_frames)} vs {len(reconstructed_frames)}")
            # Trim to shorter length
            min_len = min(len(original_frames), len(reconstructed_frames))
            original_frames = original_frames[:min_len]
            reconstructed_frames = reconstructed_frames[:min_len]
        
        psnr_values = []
        ssim_values = []
        
        logger.info(f"Calculating metrics for {len(original_frames)} frames...")
        
        for i, (orig, recon) in enumerate(zip(original_frames, reconstructed_frames)):
            metrics = QualityMetrics.calculate_frame_metrics(orig, recon)
            psnr_values.append(metrics['psnr_db'])
            ssim_values.append(metrics['ssim'])
            
            if (i + 1) % 30 == 0:
                logger.info(f"  Processed {i + 1}/{len(original_frames)} frames")
        
        return {
            'avg_psnr_db': float(np.mean(psnr_values)),
            'min_psnr_db': float(np.min(psnr_values)),
            'max_psnr_db': float(np.max(psnr_values)),
            'avg_ssim': float(np.mean(ssim_values)),
            'min_ssim': float(np.min(ssim_values)),
            'max_ssim': float(np.max(ssim_values)),
            'frame_count': len(original_frames)
        }
    
    @staticmethod
    def calculate_vmaf(original_video: str,
                      reconstructed_video: str,
                      model_path: Optional[str] = None) -> Optional[float]:
        """
        Calculate VMAF (Video Multimethod Assessment Fusion).
        
        Requires FFmpeg with libvmaf support.
        
        Args:
            original_video: Path to original video
            reconstructed_video: Path to reconstructed video
            model_path: Optional path to VMAF model file
            
        Returns:
            VMAF score (0-100, higher is better) or None if failed
        """
        import subprocess
        import tempfile
        import json
        
        try:
            # Create temp file for VMAF output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                vmaf_output = f.name
            
            # Build FFmpeg command
            cmd = [
                'ffmpeg',
                '-i', reconstructed_video,  # distorted
                '-i', original_video,       # reference
                '-lavfi',
                f'libvmaf=log_path={vmaf_output}:log_fmt=json',
                '-f', 'null',
                '-'
            ]
            
            # Run FFmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"VMAF calculation failed: {result.stderr}")
                return None
            
            # Parse VMAF output
            with open(vmaf_output, 'r') as f:
                vmaf_data = json.load(f)
            
            # Extract average VMAF score
            if 'pooled_metrics' in vmaf_data:
                vmaf_score = vmaf_data['pooled_metrics']['vmaf']['mean']
            elif 'frames' in vmaf_data:
                vmaf_scores = [frame['metrics']['vmaf'] for frame in vmaf_data['frames']]
                vmaf_score = np.mean(vmaf_scores)
            else:
                logger.error("Could not parse VMAF output")
                return None
            
            logger.info(f"✅ VMAF score: {vmaf_score:.2f}")
            return float(vmaf_score)
            
        except Exception as e:
            logger.error(f"VMAF calculation error: {e}")
            return None
    
    @staticmethod
    def load_video_frames(video_path: str,
                         max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Load all frames from a video file.
        
        Args:
            video_path: Path to video
            max_frames: Maximum frames to load (None = all)
            
        Returns:
            List of BGR frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_idx >= max_frames:
                break
            
            frames.append(frame)
            frame_idx += 1
        
        cap.release()
        logger.info(f"Loaded {len(frames)} frames from {video_path}")
        
        return frames
    
    @staticmethod
    def compare_videos(original_path: str,
                      reconstructed_path: str,
                      use_vmaf: bool = False) -> Dict:
        """
        Complete comparison of two videos.
        
        Args:
            original_path: Original video path
            reconstructed_path: Reconstructed video path
            use_vmaf: Whether to calculate VMAF (requires FFmpeg)
            
        Returns:
            Complete metrics dictionary
        """
        # Load frames
        orig_frames = QualityMetrics.load_video_frames(original_path)
        recon_frames = QualityMetrics.load_video_frames(reconstructed_path)
        
        # Calculate PSNR/SSIM
        metrics = QualityMetrics.calculate_video_metrics(orig_frames, recon_frames)
        
        # Calculate VMAF if requested
        if use_vmaf:
            vmaf_score = QualityMetrics.calculate_vmaf(original_path, reconstructed_path)
            metrics['vmaf'] = vmaf_score
        
        return metrics


# Example usage
if __name__ == "__main__":
    # Test with synthetic frames
    original = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Add noise to create reconstructed
    noise = np.random.randint(-10, 10, original.shape, dtype=np.int16)
    reconstructed = np.clip(original.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Calculate metrics
    metrics = QualityMetrics.calculate_frame_metrics(original, reconstructed)
    print(f"PSNR: {metrics['psnr_db']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    
    # Test with multiple frames
    orig_frames = [original] * 10
    recon_frames = [reconstructed] * 10
    
    video_metrics = QualityMetrics.calculate_video_metrics(orig_frames, recon_frames)
    print(f"\nVideo metrics:")
    print(f"  Avg PSNR: {video_metrics['avg_psnr_db']:.2f} dB")
    print(f"  Avg SSIM: {video_metrics['avg_ssim']:.4f}")

