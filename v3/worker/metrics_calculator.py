"""
V3.0 Metrics Calculator

Calculates video compression metrics: PSNR, SSIM, bitrate, compression ratio
"""

import os
import cv2
import numpy as np
import logging
from skimage.metrics import structural_similarity as ssim
from typing import Dict

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate video compression quality metrics"""
    
    def calculate_metrics(
        self,
        original_path: str,
        compressed_path: str,
        reconstructed_path: str
    ) -> Dict[str, float]:
        """
        Calculate all metrics for an experiment
        
        Returns:
            {
                'psnr_db': float,
                'ssim': float,
                'bitrate_mbps': float,
                'compression_ratio': float,
                'original_size_bytes': int,
                'compressed_size_bytes': int
            }
        """
        try:
            # Get file sizes
            original_size = os.path.getsize(original_path)
            compressed_size = os.path.getsize(compressed_path)
            
            # Calculate compression ratio
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            
            # Calculate bitrate
            cap = cv2.VideoCapture(reconstructed_path)
            fps = cap.get(cv2.CV_CAP_PROP_FPS) if hasattr(cv2, 'CV_CAP_PROP_FPS') else cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CV_CAP_PROP_FRAME_COUNT) if hasattr(cv2, 'CV_CAP_PROP_FRAME_COUNT') else cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            duration_sec = frame_count / fps if fps > 0 else 1
            bitrate_mbps = (compressed_size * 8) / (duration_sec * 1_000_000)
            
            # Calculate PSNR and SSIM
            psnr_db = self._calculate_psnr(original_path, reconstructed_path)
            ssim_score = self._calculate_ssim(original_path, reconstructed_path)
            
            logger.info(f"📊 Metrics calculated:")
            logger.info(f"   PSNR: {psnr_db:.2f} dB")
            logger.info(f"   SSIM: {ssim_score:.3f}")
            logger.info(f"   Bitrate: {bitrate_mbps:.2f} Mbps")
            logger.info(f"   Compression: {compression_ratio:.1f}x")
            
            return {
                'psnr_db': round(psnr_db, 2),
                'ssim': round(ssim_score, 3),
                'bitrate_mbps': round(bitrate_mbps, 3),
                'compression_ratio': round(compression_ratio, 2),
                'original_size_bytes': original_size,
                'compressed_size_bytes': compressed_size
            }
            
        except Exception as e:
            logger.error(f"❌ Metrics calculation error: {e}", exc_info=True)
            return {
                'psnr_db': 0.0,
                'ssim': 0.0,
                'bitrate_mbps': 0.0,
                'compression_ratio': 0.0,
                'original_size_bytes': 0,
                'compressed_size_bytes': 0
            }
    
    def _calculate_psnr(self, original_path: str, reconstructed_path: str) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        try:
            cap_orig = cv2.VideoCapture(original_path)
            cap_recon = cv2.VideoCapture(reconstructed_path)
            
            psnr_values = []
            
            while True:
                ret_orig, frame_orig = cap_orig.read()
                ret_recon, frame_recon = cap_recon.read()
                
                if not ret_orig or not ret_recon:
                    break
                
                # Ensure same size
                if frame_orig.shape != frame_recon.shape:
                    frame_recon = cv2.resize(frame_recon, (frame_orig.shape[1], frame_orig.shape[0]))
                
                # Calculate MSE
                mse = np.mean((frame_orig.astype(float) - frame_recon.astype(float)) ** 2)
                
                if mse == 0:
                    psnr = 100  # Perfect match
                else:
                    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                
                psnr_values.append(psnr)
            
            cap_orig.release()
            cap_recon.release()
            
            return np.mean(psnr_values) if psnr_values else 0.0
            
        except Exception as e:
            logger.error(f"PSNR calculation error: {e}")
            return 0.0
    
    def _calculate_ssim(self, original_path: str, reconstructed_path: str) -> float:
        """Calculate Structural Similarity Index"""
        try:
            cap_orig = cv2.VideoCapture(original_path)
            cap_recon = cv2.VideoCapture(reconstructed_path)
            
            ssim_values = []
            
            while True:
                ret_orig, frame_orig = cap_orig.read()
                ret_recon, frame_recon = cap_recon.read()
                
                if not ret_orig or not ret_recon:
                    break
                
                # Ensure same size
                if frame_orig.shape != frame_recon.shape:
                    frame_recon = cv2.resize(frame_recon, (frame_orig.shape[1], frame_orig.shape[0]))
                
                # Convert to grayscale for SSIM
                gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
                gray_recon = cv2.cvtColor(frame_recon, cv2.COLOR_BGR2GRAY)
                
                # Calculate SSIM
                score = ssim(gray_orig, gray_recon)
                ssim_values.append(score)
            
            cap_orig.release()
            cap_recon.release()
            
            return np.mean(ssim_values) if ssim_values else 0.0
            
        except Exception as e:
            logger.error(f"SSIM calculation error: {e}")
            return 0.0

