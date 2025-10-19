"""
Residual Encoder - PVC Module

Computes and encodes residuals (differences between procedural reconstruction and original).
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResidualEncoder:
    """
    Encodes residuals for regions where procedural reconstruction has high error.
    """
    
    def __init__(self, error_threshold: float = 10.0, tile_size: int = 64):
        """
        Initialize residual encoder.
        
        Args:
            error_threshold: MSE threshold above which to store residuals
            tile_size: Size of residual tiles (e.g., 64x64)
        """
        self.error_threshold = error_threshold
        self.tile_size = tile_size
    
    def compute_residuals(self,
                         original_frames: List[np.ndarray],
                         reconstructed_frames: List[np.ndarray]) -> List[Dict]:
        """
        Compute residuals between original and reconstructed frames.
        
        Args:
            original_frames: List of original video frames
            reconstructed_frames: List of procedurally reconstructed frames
            
        Returns:
            List of residual dictionaries for each frame
        """
        if len(original_frames) != len(reconstructed_frames):
            logger.warning(f"Frame count mismatch: {len(original_frames)} vs {len(reconstructed_frames)}")
            min_len = min(len(original_frames), len(reconstructed_frames))
            original_frames = original_frames[:min_len]
            reconstructed_frames = reconstructed_frames[:min_len]
        
        residuals = []
        
        logger.info(f"Computing residuals for {len(original_frames)} frames...")
        
        for frame_idx, (orig, recon) in enumerate(zip(original_frames, reconstructed_frames)):
            frame_residuals = self._compute_frame_residuals(orig, recon, frame_idx)
            residuals.append(frame_residuals)
            
            if (frame_idx + 1) % 30 == 0:
                logger.info(f"  Processed {frame_idx + 1}/{len(original_frames)} frames")
        
        # Count total residual tiles
        total_tiles = sum(len(r['tiles']) for r in residuals)
        logger.info(f"✅ Found {total_tiles} residual tiles across {len(residuals)} frames")
        
        return residuals
    
    def _compute_frame_residuals(self,
                                 original: np.ndarray,
                                 reconstructed: np.ndarray,
                                 frame_idx: int) -> Dict:
        """
        Compute residuals for a single frame.
        
        Args:
            original: Original frame
            reconstructed: Reconstructed frame
            frame_idx: Frame index
            
        Returns:
            Dictionary with frame residuals
        """
        # Ensure same size
        if original.shape != reconstructed.shape:
            reconstructed = cv2.resize(reconstructed, (original.shape[1], original.shape[0]))
        
        # Compute error map (MSE per pixel)
        diff = original.astype(float) - reconstructed.astype(float)
        error_map = np.mean(diff ** 2, axis=2)  # Average across color channels
        
        # Divide frame into tiles and check error
        height, width = original.shape[:2]
        tiles = []
        
        for y in range(0, height, self.tile_size):
            for x in range(0, width, self.tile_size):
                # Get tile bounds
                y1 = y
                x1 = x
                y2 = min(y + self.tile_size, height)
                x2 = min(x + self.tile_size, width)
                
                # Calculate average error in this tile
                tile_error = np.mean(error_map[y1:y2, x1:x2])
                
                # If error is above threshold, store this tile
                if tile_error > self.error_threshold:
                    # Extract residual tile (difference)
                    residual_tile = diff[y1:y2, x1:x2].astype(np.int16)
                    
                    # Compress tile using PNG (lossless)
                    # In production, would use better codec (AV1, VQ, etc.)
                    _, encoded = cv2.imencode('.png', original[y1:y2, x1:x2])
                    
                    # Convert bytes to base64 for JSON serialization
                    import base64
                    encoded_b64 = base64.b64encode(encoded.tobytes()).decode('ascii')
                    
                    tiles.append({
                        'position': [int(x1), int(y1)],
                        'size': [int(x2 - x1), int(y2 - y1)],
                        'error': float(tile_error),
                        'data': encoded_b64,  # Base64 encoded compressed tile
                        'data_size': len(encoded)
                    })
        
        return {
            'frame_idx': frame_idx,
            'tiles': tiles,
            'total_error': float(np.mean(error_map)),
            'max_error': float(np.max(error_map))
        }
    
    def estimate_residual_size(self, residuals: List[Dict]) -> Dict:
        """
        Estimate size of residual data.
        
        Args:
            residuals: List of frame residuals
            
        Returns:
            Dictionary with size estimates
        """
        total_tiles = 0
        total_bytes = 0
        
        for frame_res in residuals:
            tiles = frame_res.get('tiles', [])
            total_tiles += len(tiles)
            total_bytes += sum(t.get('data_size', 0) for t in tiles)
        
        return {
            'total_tiles': total_tiles,
            'total_bytes': total_bytes,
            'total_kb': total_bytes / 1024,
            'total_mb': total_bytes / (1024 * 1024),
            'avg_tile_size': total_bytes / total_tiles if total_tiles > 0 else 0
        }


# Example usage
if __name__ == "__main__":
    # Test with synthetic frames
    original = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Create reconstructed with some errors
    noise = np.random.randint(-20, 20, original.shape, dtype=np.int16)
    reconstructed = np.clip(original.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Compute residuals
    encoder = ResidualEncoder(error_threshold=10.0, tile_size=64)
    residuals = encoder.compute_residuals([original], [reconstructed])
    
    size_info = encoder.estimate_residual_size(residuals)
    print(f"Residual tiles: {size_info['total_tiles']}")
    print(f"Residual size: {size_info['total_kb']:.2f} KB")

