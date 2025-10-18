import numpy as np
import cv2
import struct
import json
import base64
from typing import Dict, Tuple

# Configuration
KEYFRAME_INTERVAL = 10  # Store full frame every N frames
SPATIAL_DOWNSCALE = 0.5  # Downscale to 50% for keyframes
JPEG_QUALITY = 75  # JPEG quality for keyframe compression
INTERPOLATION_METHOD = cv2.INTER_LINEAR

# Global cache for previous keyframe (used for inter-frame reconstruction)
_prev_keyframe = None
_prev_keyframe_index = -1

def compress_video_frame(frame: np.ndarray, frame_index: int, config: dict) -> bytes:
    """
    Compresses a video frame using keyframe + interpolation approach.
    
    Args:
        frame: Input frame as numpy array (H, W, 3) in BGR format
        frame_index: Frame number in sequence (0-indexed)
        config: Configuration dictionary
    
    Returns:
        Compressed frame data as bytes
    """
    global _prev_keyframe, _prev_keyframe_index
    
    # Extract config parameters
    keyframe_interval = config.get('keyframe_interval', KEYFRAME_INTERVAL)
    spatial_downscale = config.get('spatial_downscale', SPATIAL_DOWNSCALE)
    jpeg_quality = config.get('jpeg_quality', JPEG_QUALITY)
    
    is_keyframe = (frame_index % keyframe_interval == 0)
    
    # Header: 1 byte for frame type (0=keyframe, 1=interpolated) + 4 bytes for frame_index
    header = struct.pack('B I', int(is_keyframe), frame_index)
    
    if is_keyframe:
        # Keyframe: downsample and compress with JPEG
        h, w = frame.shape[:2]
        new_h, new_w = int(h * spatial_downscale), int(w * spatial_downscale)
        downscaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # JPEG compress
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, encoded = cv2.imencode('.jpg', downscaled, encode_param)
        compressed_data = encoded.tobytes()
        
        # Store dimensions for decompression
        dims = struct.pack('I I I I', h, w, new_h, new_w)
        
        # Update cache
        _prev_keyframe = downscaled.copy()
        _prev_keyframe_index = frame_index
        
        return header + dims + compressed_data
    
    else:
        # Non-keyframe: store motion/residual information
        # For simplicity, store downsampled residual from previous keyframe
        if _prev_keyframe is None:
            # Fallback: treat as keyframe if no previous keyframe exists
            return compress_video_frame(frame, frame_index, {**config, 'keyframe_interval': 1})
        
        # Downsample current frame
        h, w = frame.shape[:2]
        new_h, new_w = int(h * spatial_downscale), int(w * spatial_downscale)
        downscaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Calculate residual (difference from previous keyframe)
        # Resize prev keyframe to match if needed
        if _prev_keyframe.shape != downscaled.shape:
            prev_resized = cv2.resize(_prev_keyframe, (new_w, new_h), interpolation=INTERPOLATION_METHOD)
        else:
            prev_resized = _prev_keyframe
        
        # Compute residual
        residual = downscaled.astype(np.int16) - prev_resized.astype(np.int16)
        
        # Quantize residual to reduce size (simple uniform quantization)
        quantization_step = 8  # Increase to reduce bitrate, decrease for quality
        quantized_residual = (residual // quantization_step).astype(np.int8)
        
        # Compress quantized residual with zlib-like encoding (PNG for simplicity)
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
        # Convert int8 residual to uint8 for encoding (shift by 128)
        residual_uint8 = (quantized_residual + 128).astype(np.uint8)
        _, encoded = cv2.imencode('.png', residual_uint8, encode_param)
        compressed_data = encoded.tobytes()
        
        # Store metadata
        dims = struct.pack('I I I I B', h, w, new_h, new_w, quantization_step)
        
        return header + dims + compressed_data


def decompress_video_frame(compressed_data: bytes, frame_index: int, config: dict) -> np.ndarray:
    """
    Decompresses a video frame from compressed bytes.
    
    Args:
        compressed_data: Compressed frame data
        frame_index: Frame number in sequence
        config: Configuration dictionary
    
    Returns:
        Reconstructed frame as numpy array (H, W, 3) in BGR format
    """
    global _prev_keyframe, _prev_keyframe_index
    
    # Parse header
    is_keyframe = bool(struct.unpack('B', compressed_data[0:1])[0])
    stored_frame_index = struct.unpack('I', compressed_data[1:5])[0]
    
    if is_keyframe:
        # Parse dimensions
        h, w, new_h, new_w = struct.unpack('I I I I', compressed_data[5:21])
        
        # Decode JPEG
        jpeg_data = compressed_data[21:]
        nparr = np.frombuffer(jpeg_data, np.uint8)
        downscaled = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if downscaled is None:
            raise ValueError(f"Failed to decode keyframe {frame_index}")
        
        # Upscale to original resolution
        reconstructed = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Update cache
        _prev_keyframe = downscaled.copy()
        _prev_keyframe_index = frame_index
        
        return reconstructed
    
    else:
        # Parse dimensions and quantization step
        h, w, new_h, new_w, quantization_step = struct.unpack('I I I I B', compressed_data[5:22])
        
        # Decode residual
        residual_data = compressed_data[22:]
        nparr = np.frombuffer(residual_data, np.uint8)
        residual_uint8 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if residual_uint8 is None:
            raise ValueError(f"Failed to decode residual for frame {frame_index}")
        
        # Convert back to int8 (unshift)
        quantized_residual = residual_uint8.astype(np.int16) - 128
        
        # Dequantize
        residual = quantized_residual * quantization_step
        
        # Add to previous keyframe
        if _prev_keyframe is None:
            raise ValueError(f"No previous keyframe available for frame {frame_index}")
        
        # Ensure sizes match
        if _prev_keyframe.shape[:2] != (new_h, new_w):
            prev_resized = cv2.resize(_prev_keyframe, (new_w, new_h), interpolation=INTERPOLATION_METHOD)
        else:
            prev_resized = _prev_keyframe
        
        # Reconstruct downscaled frame
        reconstructed_downscaled = np.clip(prev_resized.astype(np.int16) + residual, 0, 255).astype(np.uint8)
        
        # Upscale to original resolution
        reconstructed = cv2.resize(reconstructed_downscaled, (w, h), interpolation=cv2.INTER_CUBIC)
        
        return reconstructed


# Example usage and testing
if __name__ == "__main__":
    # Test with synthetic frame
    test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    config = {
        'keyframe_interval': 10,
        'spatial_downscale': 0.5,
        'jpeg_quality': 75
    }
    
    # Compress and decompress
    compressed = compress_video_frame(test_frame, 0, config)
    reconstructed = decompress_video_frame(compressed, 0, config)
    
    print(f"Original size: {test_frame.nbytes} bytes")
    print(f"Compressed size: {len(compressed)} bytes")
    print(f"Compression ratio: {len(compressed) / test_frame.nbytes * 100:.2f}%")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Calculate PSNR
    mse = np.mean((test_frame.astype(float) - reconstructed.astype(float)) ** 2)
    if mse > 0:
        psnr = 10 * np.log10(255**2 / mse)
        print(f"PSNR: {psnr:.2f} dB")
    else:
        print("PSNR: Infinite (perfect reconstruction)")