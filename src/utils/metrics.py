#!/usr/bin/env python3
"""
Metrics Calculator for AI Video Codec Framework
Calculates PSNR, SSIM, VMAF, and other video quality metrics.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Optional
import subprocess
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate video quality metrics."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def calculate_psnr(self, reference_path: str, compressed_path: str) -> float:
        """Calculate Peak Signal-to-Noise Ratio (PSNR) between two videos."""
        try:
            # Load videos
            cap_ref = cv2.VideoCapture(reference_path)
            cap_comp = cv2.VideoCapture(compressed_path)
            
            if not cap_ref.isOpened() or not cap_comp.isOpened():
                logger.error("Could not open video files")
                return 0.0
            
            psnr_values = []
            frame_count = 0
            
            while True:
                ret_ref, frame_ref = cap_ref.read()
                ret_comp, frame_comp = cap_comp.read()
                
                if not ret_ref or not ret_comp:
                    break
                
                # Ensure same dimensions
                if frame_ref.shape != frame_comp.shape:
                    frame_comp = cv2.resize(frame_comp, (frame_ref.shape[1], frame_ref.shape[0]))
                
                # Calculate PSNR for this frame
                mse = np.mean((frame_ref.astype(np.float32) - frame_comp.astype(np.float32)) ** 2)
                if mse == 0:
                    psnr = 100.0  # Perfect match
                else:
                    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                
                psnr_values.append(psnr)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames for PSNR calculation")
            
            cap_ref.release()
            cap_comp.release()
            
            if not psnr_values:
                logger.error("No frames processed for PSNR calculation")
                return 0.0
            
            avg_psnr = np.mean(psnr_values)
            logger.info(f"Average PSNR: {avg_psnr:.2f} dB")
            return avg_psnr
            
        except Exception as e:
            logger.error(f"Error calculating PSNR: {e}")
            return 0.0
    
    def calculate_ssim(self, reference_path: str, compressed_path: str) -> float:
        """Calculate Structural Similarity Index (SSIM) between two videos."""
        try:
            # Load videos
            cap_ref = cv2.VideoCapture(reference_path)
            cap_comp = cv2.VideoCapture(compressed_path)
            
            if not cap_ref.isOpened() or not cap_comp.isOpened():
                logger.error("Could not open video files")
                return 0.0
            
            ssim_values = []
            frame_count = 0
            
            while True:
                ret_ref, frame_ref = cap_ref.read()
                ret_comp, frame_comp = cap_comp.read()
                
                if not ret_ref or not ret_comp:
                    break
                
                # Ensure same dimensions
                if frame_ref.shape != frame_comp.shape:
                    frame_comp = cv2.resize(frame_comp, (frame_ref.shape[1], frame_ref.shape[0]))
                
                # Convert to grayscale for SSIM
                gray_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
                gray_comp = cv2.cvtColor(frame_comp, cv2.COLOR_BGR2GRAY)
                
                # Calculate SSIM for this frame
                ssim = self._calculate_frame_ssim(gray_ref, gray_comp)
                ssim_values.append(ssim)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames for SSIM calculation")
            
            cap_ref.release()
            cap_comp.release()
            
            if not ssim_values:
                logger.error("No frames processed for SSIM calculation")
                return 0.0
            
            avg_ssim = np.mean(ssim_values)
            logger.info(f"Average SSIM: {avg_ssim:.4f}")
            return avg_ssim
            
        except Exception as e:
            logger.error(f"Error calculating SSIM: {e}")
            return 0.0
    
    def _calculate_frame_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM for a single frame."""
        # Constants
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # Convert to float
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        # Calculate means
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return np.mean(ssim_map)
    
    def calculate_vmaf(self, reference_path: str, compressed_path: str) -> float:
        """Calculate VMAF (Video Multi-Method Assessment Fusion) score."""
        try:
            # Check if ffmpeg is available
            if not self._check_ffmpeg():
                logger.warning("FFmpeg not available, skipping VMAF calculation")
                return 0.0
            
            # Create temporary files for VMAF calculation
            ref_yuv = os.path.join(self.temp_dir, "reference.yuv")
            comp_yuv = os.path.join(self.temp_dir, "compressed.yuv")
            vmaf_output = os.path.join(self.temp_dir, "vmaf.json")
            
            # Convert videos to YUV format
            self._convert_to_yuv(reference_path, ref_yuv)
            self._convert_to_yuv(compressed_path, comp_yuv)
            
            # Calculate VMAF using ffmpeg
            cmd = [
                'ffmpeg', '-i', compressed_path, '-i', reference_path,
                '-lavfi', 'libvmaf=model_path=/usr/local/share/model/vmaf_v0.6.1.pkl:log_path=' + vmaf_output,
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"VMAF calculation failed: {result.stderr}")
                return 0.0
            
            # Parse VMAF score from output
            vmaf_score = self._parse_vmaf_score(vmaf_output)
            
            logger.info(f"VMAF score: {vmaf_score:.2f}")
            return vmaf_score
            
        except Exception as e:
            logger.error(f"Error calculating VMAF: {e}")
            return 0.0
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _convert_to_yuv(self, input_path: str, output_path: str):
        """Convert video to YUV format."""
        cmd = [
            'ffmpeg', '-i', input_path, '-pix_fmt', 'yuv420p', '-y', output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
    
    def _parse_vmaf_score(self, vmaf_output_path: str) -> float:
        """Parse VMAF score from JSON output."""
        try:
            import json
            with open(vmaf_output_path, 'r') as f:
                data = json.load(f)
                return float(data['pooled_metrics']['vmaf']['mean'])
        except Exception as e:
            logger.error(f"Error parsing VMAF score: {e}")
            return 0.0
    
    def calculate_bitrate(self, video_path: str) -> float:
        """Calculate video bitrate in Mbps."""
        try:
            file_size = os.path.getsize(video_path)
            
            # Get video duration using ffprobe
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Could not get video duration: {result.stderr}")
                return 0.0
            
            duration = float(result.stdout.strip())
            bitrate_bps = (file_size * 8) / duration
            bitrate_mbps = bitrate_bps / 1_000_000
            
            logger.info(f"Bitrate: {bitrate_mbps:.2f} Mbps")
            return bitrate_mbps
            
        except Exception as e:
            logger.error(f"Error calculating bitrate: {e}")
            return 0.0
    
    def calculate_compression_ratio(self, original_path: str, compressed_path: str) -> float:
        """Calculate compression ratio."""
        try:
            original_size = os.path.getsize(original_path)
            compressed_size = os.path.getsize(compressed_path)
            
            ratio = compressed_size / original_size
            logger.info(f"Compression ratio: {ratio:.4f} ({compressed_size}/{original_size})")
            return ratio
            
        except Exception as e:
            logger.error(f"Error calculating compression ratio: {e}")
            return 0.0
    
    def calculate_all_metrics(self, reference_path: str, compressed_path: str) -> Dict:
        """Calculate all quality metrics."""
        logger.info("Calculating all quality metrics...")
        
        metrics = {
            'psnr_db': self.calculate_psnr(reference_path, compressed_path),
            'ssim': self.calculate_ssim(reference_path, compressed_path),
            'vmaf': self.calculate_vmaf(reference_path, compressed_path),
            'bitrate_mbps': self.calculate_bitrate(compressed_path),
            'compression_ratio': self.calculate_compression_ratio(reference_path, compressed_path),
        }
        
        # Calculate quality score (weighted combination)
        quality_score = (
            metrics['psnr_db'] * 0.4 +
            metrics['ssim'] * 100 * 0.3 +  # Scale SSIM to 0-100 range
            metrics['vmaf'] * 0.3
        )
        metrics['quality_score'] = quality_score
        
        logger.info(f"All metrics calculated: {metrics}")
        return metrics
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Could not clean up temp directory {self.temp_dir}: {e}")
