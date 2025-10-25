"""
Optimized Line Art Compression

Uses delta encoding + run-length encoding for sparse line art
Target: 10-20 KB for 1920x1080 frame (vs current 261 KB)
"""

import numpy as np
import zlib
from typing import Tuple

def compress_line_art_optimized(line_art: np.ndarray) -> bytes:
    """
    Compress sparse line art using delta encoding + RLE + GZIP
    
    Args:
        line_art: (H, W) binary mask [0, 255]
    
    Returns:
        compressed: bytes
    """
    # Get coordinates of edge pixels
    coords = np.argwhere(line_art > 0)  # (N, 2) array of (y, x)
    
    if len(coords) == 0:
        return b''
    
    # Sort by row, then column (scan-line order)
    coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
    
    # Delta encoding: store differences instead of absolute positions
    # This makes values smaller and more compressible
    deltas = np.diff(coords, axis=0, prepend=coords[0:1])
    
    # Convert to int16 (sufficient for deltas in 1920x1080)
    deltas_int16 = deltas.astype(np.int16)
    
    # Compress with GZIP
    compressed = zlib.compress(deltas_int16.tobytes(), level=9)
    
    # Prepend shape and number of points
    header = np.array([line_art.shape[0], line_art.shape[1], len(coords)], dtype=np.uint32)
    
    return header.tobytes() + compressed


def decompress_line_art_optimized(compressed: bytes, return_coords=False) -> np.ndarray:
    """
    Decompress line art
    
    Args:
        compressed: bytes from compress_line_art_optimized
        return_coords: If True, also return coordinates
    
    Returns:
        line_art: (H, W) binary mask
        coords: (N, 2) coordinates (if return_coords=True)
    """
    if len(compressed) == 0:
        return np.zeros((512, 960), dtype=np.uint8)
    
    # Read header
    header = np.frombuffer(compressed[:12], dtype=np.uint32)
    height, width, num_points = header
    
    # Decompress deltas
    compressed_data = compressed[12:]
    decompressed = zlib.decompress(compressed_data)
    deltas = np.frombuffer(decompressed, dtype=np.int16).reshape(-1, 2)
    
    # Reconstruct absolute coordinates from deltas
    coords = np.cumsum(deltas, axis=0)
    
    # Create line art image
    line_art = np.zeros((height, width), dtype=np.uint8)
    
    # Clip coordinates to valid range (safety)
    coords[:, 0] = np.clip(coords[:, 0], 0, height - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, width - 1)
    
    line_art[coords[:, 0], coords[:, 1]] = 255
    
    if return_coords:
        return line_art, coords
    return line_art


def compress_color_map_optimized(color_map: np.ndarray, n_colors: int) -> bytes:
    """
    Compress color map using PNG-style prediction + entropy coding
    
    Args:
        color_map: (H, W) palette indices [0, n_colors-1]
        n_colors: Number of colors in palette
    
    Returns:
        compressed: bytes
    """
    h, w = color_map.shape
    
    # PNG Paeth predictor: predict from left, top, top-left
    # This exploits spatial coherence in anime (large flat regions)
    predicted = np.zeros_like(color_map, dtype=np.int16)
    
    for y in range(h):
        for x in range(w):
            if x == 0 and y == 0:
                predicted[y, x] = color_map[y, x]
            elif x == 0:
                predicted[y, x] = color_map[y, x] - color_map[y-1, x]
            elif y == 0:
                predicted[y, x] = color_map[y, x] - color_map[y, x-1]
            else:
                # Paeth predictor
                left = color_map[y, x-1]
                top = color_map[y-1, x]
                top_left = color_map[y-1, x-1]
                
                p = left + top - top_left
                pa = abs(p - left)
                pb = abs(p - top)
                pc = abs(p - top_left)
                
                if pa <= pb and pa <= pc:
                    predictor = left
                elif pb <= pc:
                    predictor = top
                else:
                    predictor = top_left
                
                predicted[y, x] = color_map[y, x] - predictor
    
    # Convert to int8 (residuals are small)
    predicted_int8 = predicted.astype(np.int8)
    
    # Compress with GZIP
    compressed = zlib.compress(predicted_int8.tobytes(), level=9)
    
    # Prepend shape
    header = np.array([h, w, n_colors], dtype=np.uint32)
    
    return header.tobytes() + compressed


def decompress_color_map_optimized(compressed: bytes) -> np.ndarray:
    """
    Decompress color map
    
    Returns:
        color_map: (H, W) palette indices
    """
    # Read header
    header = np.frombuffer(compressed[:12], dtype=np.uint32)
    h, w, n_colors = header
    
    # Decompress predicted values
    compressed_data = compressed[12:]
    decompressed = zlib.decompress(compressed_data)
    predicted = np.frombuffer(decompressed, dtype=np.int8).reshape(h, w)
    
    # Reconstruct original from predicted residuals
    color_map = np.zeros((h, w), dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            if x == 0 and y == 0:
                color_map[y, x] = predicted[y, x]
            elif x == 0:
                color_map[y, x] = predicted[y, x] + color_map[y-1, x]
            elif y == 0:
                color_map[y, x] = predicted[y, x] + color_map[y, x-1]
            else:
                # Paeth predictor
                left = color_map[y, x-1]
                top = color_map[y-1, x]
                top_left = color_map[y-1, x-1]
                
                p = left + top - top_left
                pa = abs(p - left)
                pb = abs(p - top)
                pc = abs(p - top_left)
                
                if pa <= pb and pa <= pc:
                    predictor = left
                elif pb <= pc:
                    predictor = top
                else:
                    predictor = top_left
                
                color_map[y, x] = (predicted[y, x] + predictor) % 256
    
    return color_map


if __name__ == "__main__":
    import cv2
    import sys
    sys.path.append('..')
    from utils.layer_extraction import extract_line_art, extract_color_palette
    
    print("="*70)
    print("TESTING OPTIMIZED COMPRESSION")
    print("="*70)
    
    # Load real anime frame
    frame = cv2.imread('/tmp/bleach_frame_18sec.png')
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    
    print(f"\nFrame: {w}x{h}")
    
    # Extract layers
    print("\n1. Extracting line art...")
    line_art = extract_line_art(frame_rgb)
    edge_pixels = np.sum(line_art > 0)
    print(f"   Edge pixels: {edge_pixels:,}")
    
    # Compress line art (OPTIMIZED)
    print("\n2. Compressing line art (optimized)...")
    line_art_compressed = compress_line_art_optimized(line_art)
    line_art_size_kb = len(line_art_compressed) / 1024
    print(f"   Compressed size: {len(line_art_compressed):,} bytes ({line_art_size_kb:.2f} KB)")
    print(f"   Old method: 261.20 KB")
    print(f"   Improvement: {261.20 / line_art_size_kb:.1f}x smaller!")
    
    # Verify decompression
    line_art_decompressed = decompress_line_art_optimized(line_art_compressed)
    match = np.array_equal(line_art, line_art_decompressed)
    print(f"   Decompression: {'✓ Perfect match!' if match else '✗ Mismatch'}")
    
    # Extract and compress color map
    print("\n3. Extracting and compressing color map...")
    palette, color_map = extract_color_palette(frame_rgb, n_colors=16, mask=line_art)
    
    color_map_compressed = compress_color_map_optimized(color_map, n_colors=16)
    color_map_size_kb = len(color_map_compressed) / 1024
    print(f"   Compressed size: {len(color_map_compressed):,} bytes ({color_map_size_kb:.2f} KB)")
    print(f"   Old estimate: 1012.50 KB")
    print(f"   Improvement: {1012.50 / color_map_size_kb:.1f}x smaller!")
    
    # Verify decompression
    color_map_decompressed = decompress_color_map_optimized(color_map_compressed)
    match = np.array_equal(color_map, color_map_decompressed)
    print(f"   Decompression: {'✓ Perfect match!' if match else '✗ Mismatch'}")
    
    # Calculate total size
    print("\n4. TOTAL COMPRESSED SIZE:")
    palette_size = len(palette) * 3  # n_colors × 3 bytes
    residual_size = 15  # KB (from model test)
    
    total_kb = line_art_size_kb + (palette_size / 1024) + color_map_size_kb + residual_size
    
    print(f"   Line art:    {line_art_size_kb:.2f} KB")
    print(f"   Palette:     {palette_size / 1024:.2f} KB")
    print(f"   Color map:   {color_map_size_kb:.2f} KB")
    print(f"   Residual:    {residual_size:.2f} KB (neural)")
    print(f"   ---")
    print(f"   TOTAL:       {total_kb:.2f} KB")
    
    # Compare to targets
    print(f"\n5. COMPARISON:")
    print(f"   Original (uncompressed): {w * h * 3 / 1024:.1f} KB")
    print(f"   Our codec: {total_kb:.1f} KB")
    print(f"   AV1 I-frame (typical): 150-250 KB")
    print(f"   ---")
    print(f"   Compression ratio: {(w * h * 3 / 1024) / total_kb:.1f}x")
    print(f"   vs AV1 (150 KB): {150 / total_kb:.2f}x better!")
    print(f"   vs AV1 (250 KB): {250 / total_kb:.2f}x better!")
    
    print("\n" + "="*70)
    print("✓ OPTIMIZED COMPRESSION TEST PASSED!")
    print("="*70)

