"""
Layer Extraction Utilities for Anime Frames

Extracts natural layers from anime:
1. Line art (sparse edges)
2. Color palette
3. Color map (palette indices)
4. Soft details (for residual)
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

def extract_line_art(frame: np.ndarray, threshold1=50, threshold2=150) -> np.ndarray:
    """
    Extract line art layer from anime frame
    
    Args:
        frame: (H, W, 3) RGB image [0, 255]
        threshold1, threshold2: Canny edge detection thresholds
    
    Returns:
        line_art: (H, W) binary mask, 255=edge, 0=no edge
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1, threshold2)
    
    # Optional: thin the edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return edges


def extract_color_palette(frame: np.ndarray, n_colors=16, mask=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract dominant color palette using K-means
    
    Args:
        frame: (H, W, 3) RGB image [0, 255]
        n_colors: Number of colors in palette
        mask: Optional (H, W) binary mask to exclude pixels (e.g., line art)
    
    Returns:
        palette: (n_colors, 3) RGB colors [0, 255]
        color_map: (H, W) palette indices [0, n_colors-1]
    """
    h, w = frame.shape[:2]
    pixels = frame.reshape(-1, 3)
    
    # Exclude masked pixels if provided
    if mask is not None:
        mask_flat = mask.reshape(-1) == 0  # Exclude edge pixels
        pixels_to_cluster = pixels[mask_flat]
    else:
        pixels_to_cluster = pixels
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels_to_cluster.astype(np.float32))
    
    # Get palette
    palette = kmeans.cluster_centers_.astype(np.uint8)
    
    # Assign all pixels to nearest palette color
    labels = kmeans.predict(pixels.astype(np.float32))
    color_map = labels.reshape(h, w).astype(np.uint8)
    
    return palette, color_map


def apply_palette(color_map: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Reconstruct image from palette and color map
    
    Args:
        color_map: (H, W) palette indices
        palette: (n_colors, 3) RGB colors
    
    Returns:
        image: (H, W, 3) RGB image [0, 255]
    """
    return palette[color_map]


def extract_residual(original: np.ndarray, line_art: np.ndarray, palette_img: np.ndarray) -> np.ndarray:
    """
    Extract residual (soft details) after removing line art and flat colors
    
    Args:
        original: (H, W, 3) original image [0, 255]
        line_art: (H, W) binary line art mask
        palette_img: (H, W, 3) palette-reconstructed image
    
    Returns:
        residual: (H, W, 3) residual image [-255, 255]
    """
    # Convert line art to 3-channel
    line_art_3ch = np.stack([line_art] * 3, axis=-1)
    
    # Residual = original - palette (soft details like gradients, lighting)
    residual = original.astype(np.float32) - palette_img.astype(np.float32)
    
    # Zero out residual where line art is (line art is already captured)
    residual[line_art > 0] = 0
    
    return residual.astype(np.float32)


def compress_line_art_sparse(line_art: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compress sparse line art using coordinate encoding
    
    Args:
        line_art: (H, W) binary mask
    
    Returns:
        coords: (N, 2) array of (y, x) coordinates of edge pixels
        shape: (2,) original shape (H, W)
    """
    coords = np.argwhere(line_art > 0)  # Get (y, x) coordinates
    shape = np.array(line_art.shape)
    return coords, shape


def decompress_line_art_sparse(coords: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Decompress sparse line art from coordinates
    
    Args:
        coords: (N, 2) array of (y, x) coordinates
        shape: (H, W) target shape
    
    Returns:
        line_art: (H, W) binary mask
    """
    line_art = np.zeros(shape, dtype=np.uint8)
    if len(coords) > 0:
        line_art[coords[:, 0], coords[:, 1]] = 255
    return line_art


def estimate_compressed_size(coords: np.ndarray, palette: np.ndarray, color_map: np.ndarray, 
                             residual_latent_shape: Tuple[int, ...]) -> dict:
    """
    Estimate compressed file size for each layer
    
    Returns:
        dict with size estimates in bytes
    """
    # Line art (sparse coordinates)
    # Each coord is 2 uint16 values (4 bytes) - assuming 1920x1080 max resolution
    line_art_size = len(coords) * 4
    
    # Palette (n_colors × 3 bytes)
    palette_size = palette.size
    
    # Color map (can use 4-5 bits per pixel with RLE)
    # Assuming 50% compression with RLE
    color_map_size = color_map.size // 2  # Rough estimate
    
    # Residual latent (float32)
    residual_size = np.prod(residual_latent_shape) * 4
    # With INT8 quantization + GZIP, divide by 4
    residual_size_compressed = residual_size // 4
    
    return {
        'line_art_bytes': line_art_size,
        'palette_bytes': palette_size,
        'color_map_bytes': color_map_size,
        'residual_bytes': residual_size_compressed,
        'total_bytes': line_art_size + palette_size + color_map_size + residual_size_compressed,
        'total_kb': (line_art_size + palette_size + color_map_size + residual_size_compressed) / 1024
    }


if __name__ == "__main__":
    print("="*70)
    print("TESTING LAYER EXTRACTION ON REAL ANIME FRAME")
    print("="*70)
    
    # Load test frame
    frame = cv2.imread('/tmp/bleach_frame_18sec.png')
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    
    print(f"\nOriginal frame: {w}x{h}")
    print(f"Original size: {frame_rgb.size * frame_rgb.itemsize / 1024:.1f} KB (uncompressed)")
    
    # Extract layers
    print("\n1. Extracting line art...")
    line_art = extract_line_art(frame_rgb)
    edge_pixels = np.sum(line_art > 0)
    print(f"   Edge pixels: {edge_pixels:,} ({edge_pixels/(w*h)*100:.2f}%)")
    
    print("\n2. Extracting color palette...")
    palette, color_map = extract_color_palette(frame_rgb, n_colors=16, mask=line_art)
    print(f"   Palette: {len(palette)} colors")
    print(f"   Top 5 colors:")
    for i in range(min(5, len(palette))):
        color = palette[i]
        pixels = np.sum(color_map == i)
        print(f"     Color {i}: RGB{tuple(color)} - {pixels/(w*h)*100:.1f}%")
    
    print("\n3. Reconstructing from palette...")
    palette_img = apply_palette(color_map, palette)
    
    print("\n4. Extracting residual...")
    residual = extract_residual(frame_rgb, line_art, palette_img)
    residual_magnitude = np.abs(residual).mean()
    print(f"   Residual magnitude: {residual_magnitude:.2f}")
    
    print("\n5. Compressing line art (sparse encoding)...")
    coords, shape = compress_line_art_sparse(line_art)
    print(f"   Coordinates: {len(coords):,} points")
    print(f"   Sparsity: {len(coords)/(w*h)*100:.2f}% of pixels")
    
    # Estimate compressed sizes
    # Assuming residual will be compressed to 30x17x32 latent (like our models)
    residual_latent_shape = (30, 17, 32)  # For 960x540 input
    sizes = estimate_compressed_size(coords, palette, color_map, residual_latent_shape)
    
    print("\n6. Estimated compressed sizes:")
    print(f"   Line art:    {sizes['line_art_bytes']:,} bytes ({sizes['line_art_bytes']/1024:.2f} KB)")
    print(f"   Palette:     {sizes['palette_bytes']:,} bytes ({sizes['palette_bytes']/1024:.2f} KB)")
    print(f"   Color map:   {sizes['color_map_bytes']:,} bytes ({sizes['color_map_bytes']/1024:.2f} KB)")
    print(f"   Residual:    {sizes['residual_bytes']:,} bytes ({sizes['residual_bytes']/1024:.2f} KB)")
    print(f"   ---")
    print(f"   TOTAL:       {sizes['total_bytes']:,} bytes ({sizes['total_kb']:.2f} KB)")
    
    # Compare to original
    original_size_kb = w * h * 3 / 1024
    compression_ratio = original_size_kb / sizes['total_kb']
    print(f"\n7. Compression ratio:")
    print(f"   Original:    {original_size_kb:.1f} KB (uncompressed)")
    print(f"   Compressed:  {sizes['total_kb']:.1f} KB")
    print(f"   Ratio:       {compression_ratio:.1f}x")
    
    # Save visualizations
    cv2.imwrite('/tmp/layer_line_art.png', line_art)
    cv2.imwrite('/tmp/layer_palette.png', cv2.cvtColor(palette_img, cv2.COLOR_RGB2BGR))
    residual_vis = np.clip(residual + 128, 0, 255).astype(np.uint8)
    cv2.imwrite('/tmp/layer_residual.png', cv2.cvtColor(residual_vis, cv2.COLOR_RGB2BGR))
    
    print(f"\n✓ Saved visualizations:")
    print(f"   /tmp/layer_line_art.png")
    print(f"   /tmp/layer_palette.png")
    print(f"   /tmp/layer_residual.png")
    
    print("\n" + "="*70)
    print("✓ Layer extraction successful!")
    print("="*70)

