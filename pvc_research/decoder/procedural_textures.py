"""
Procedural Textures - PVC Decoder Module

Generates textures using procedural noise functions (Perlin, Worley, fBM).
Inspired by demoscene shader techniques.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProceduralTextures:
    """
    Generates procedural textures from compact parameters.
    
    Implements:
    - Perlin noise
    - Worley (cellular) noise
    - Fractional Brownian Motion (fBM)
    - Texture composition
    """
    
    @staticmethod
    def perlin_noise_2d(shape: Tuple[int, int],
                       scale: float = 10.0,
                       octaves: int = 4,
                       persistence: float = 0.5,
                       lacunarity: float = 2.0,
                       seed: int = 0) -> np.ndarray:
        """
        Generate 2D Perlin noise.
        
        Args:
            shape: (height, width) of output
            scale: Base frequency scale
            octaves: Number of noise layers
            persistence: Amplitude decay per octave
            lacunarity: Frequency increase per octave
            seed: Random seed for reproducibility
            
        Returns:
            Noise array in range [0, 1]
        """
        np.random.seed(seed)
        
        height, width = shape
        noise = np.zeros((height, width), dtype=np.float32)
        
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            # Generate gradient grid
            grid_h = int(height / scale * frequency) + 2
            grid_w = int(width / scale * frequency) + 2
            
            # Random gradients at grid points
            gradients = np.random.randn(grid_h, grid_w, 2)
            gradients /= np.linalg.norm(gradients, axis=2, keepdims=True) + 1e-8
            
            # Interpolate
            octave_noise = ProceduralTextures._interpolate_perlin(
                shape, gradients, scale * frequency
            )
            
            noise += octave_noise * amplitude
            max_value += amplitude
            
            amplitude *= persistence
            frequency *= lacunarity
        
        # Normalize to [0, 1]
        noise = (noise / max_value + 1.0) / 2.0
        return np.clip(noise, 0, 1)
    
    @staticmethod
    def _interpolate_perlin(shape: Tuple[int, int],
                           gradients: np.ndarray,
                           scale: float) -> np.ndarray:
        """Helper function to interpolate Perlin noise from gradients."""
        height, width = shape
        
        # Create coordinate grids
        y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        x_coords = x_grid / scale
        y_coords = y_grid / scale
        
        # Grid cell coordinates
        x0 = np.floor(x_coords).astype(int)
        y0 = np.floor(y_coords).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Fractional positions
        fx = x_coords - x0
        fy = y_coords - y0
        
        # Clamp grid indices
        x0 = np.clip(x0, 0, gradients.shape[1] - 1)
        x1 = np.clip(x1, 0, gradients.shape[1] - 1)
        y0 = np.clip(y0, 0, gradients.shape[0] - 1)
        y1 = np.clip(y1, 0, gradients.shape[0] - 1)
        
        # Gradient vectors at corners
        g00 = gradients[y0, x0]
        g10 = gradients[y0, x1]
        g01 = gradients[y1, x0]
        g11 = gradients[y1, x1]
        
        # Distance vectors to corners - now fx and fy have same shape
        d00 = np.stack([fx, fy], axis=-1)
        d10 = np.stack([fx - 1, fy], axis=-1)
        d01 = np.stack([fx, fy - 1], axis=-1)
        d11 = np.stack([fx - 1, fy - 1], axis=-1)
        
        # Dot products
        n00 = np.sum(g00 * d00, axis=-1)
        n10 = np.sum(g10 * d10, axis=-1)
        n01 = np.sum(g01 * d01, axis=-1)
        n11 = np.sum(g11 * d11, axis=-1)
        
        # Smooth interpolation (6t^5 - 15t^4 + 10t^3)
        u = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
        v = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
        
        # Bilinear interpolation
        nx0 = n00 * (1 - u) + n10 * u
        nx1 = n01 * (1 - u) + n11 * u
        result = nx0 * (1 - v) + nx1 * v
        
        return result
    
    @staticmethod
    def worley_noise_2d(shape: Tuple[int, int],
                       num_points: int = 20,
                       distance_func: str = 'euclidean',
                       seed: int = 0) -> np.ndarray:
        """
        Generate 2D Worley (cellular) noise.
        
        Args:
            shape: (height, width) of output
            num_points: Number of feature points
            distance_func: 'euclidean', 'manhattan', or 'chebyshev'
            seed: Random seed
            
        Returns:
            Noise array in range [0, 1]
        """
        np.random.seed(seed)
        
        height, width = shape
        
        # Generate random feature points (tiled for seamless wrapping)
        points = np.random.rand(num_points, 2)
        points[:, 0] *= height
        points[:, 1] *= width
        
        # Create coordinate grids
        y_grid, x_grid = np.meshgrid(
            np.arange(height),
            np.arange(width),
            indexing='ij'
        )
        
        # Compute distance to nearest point
        min_distances = np.full((height, width), float('inf'))
        
        for point in points:
            py, px = point
            
            if distance_func == 'euclidean':
                distances = np.sqrt((y_grid - py)**2 + (x_grid - px)**2)
            elif distance_func == 'manhattan':
                distances = np.abs(y_grid - py) + np.abs(x_grid - px)
            else:  # chebyshev
                distances = np.maximum(np.abs(y_grid - py), np.abs(x_grid - px))
            
            min_distances = np.minimum(min_distances, distances)
        
        # Normalize to [0, 1]
        max_dist = np.max(min_distances)
        if max_dist > 0:
            min_distances /= max_dist
        
        return min_distances.astype(np.float32)
    
    @staticmethod
    def fbm_noise_2d(shape: Tuple[int, int],
                    base_scale: float = 10.0,
                    octaves: int = 6,
                    persistence: float = 0.5,
                    lacunarity: float = 2.0,
                    seed: int = 0) -> np.ndarray:
        """
        Generate fractional Brownian motion (fBM) texture.
        
        This is essentially multi-octave Perlin noise with specific parameters
        tuned for naturalistic patterns.
        
        Args:
            shape: (height, width)
            base_scale: Starting frequency scale
            octaves: Number of layers
            persistence: Amplitude decay
            lacunarity: Frequency increase
            seed: Random seed
            
        Returns:
            fBM texture in [0, 1]
        """
        return ProceduralTextures.perlin_noise_2d(
            shape=shape,
            scale=base_scale,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            seed=seed
        )
    
    @staticmethod
    def generate_texture(texture_type: str,
                        shape: Tuple[int, int],
                        params: Dict) -> np.ndarray:
        """
        Generate a procedural texture from parameters.
        
        Args:
            texture_type: 'perlin', 'worley', 'fbm', or 'solid'
            shape: (height, width)
            params: Dictionary of texture parameters
            
        Returns:
            Grayscale texture array in [0, 1]
        """
        if texture_type == 'solid':
            # Solid color
            value = params.get('value', 0.5)
            return np.full(shape, value, dtype=np.float32)
        
        elif texture_type == 'perlin':
            return ProceduralTextures.perlin_noise_2d(
                shape=shape,
                scale=params.get('scale', 10.0),
                octaves=params.get('octaves', 4),
                persistence=params.get('persistence', 0.5),
                lacunarity=params.get('lacunarity', 2.0),
                seed=params.get('seed', 0)
            )
        
        elif texture_type == 'worley':
            return ProceduralTextures.worley_noise_2d(
                shape=shape,
                num_points=params.get('num_points', 20),
                distance_func=params.get('distance_func', 'euclidean'),
                seed=params.get('seed', 0)
            )
        
        elif texture_type == 'fbm':
            return ProceduralTextures.fbm_noise_2d(
                shape=shape,
                base_scale=params.get('base_scale', 10.0),
                octaves=params.get('octaves', 6),
                persistence=params.get('persistence', 0.5),
                lacunarity=params.get('lacunarity', 2.0),
                seed=params.get('seed', 0)
            )
        
        else:
            logger.warning(f"Unknown texture type: {texture_type}, using solid")
            return np.full(shape, 0.5, dtype=np.float32)
    
    @staticmethod
    def colorize_texture(texture: np.ndarray,
                        color_map: str = 'grayscale',
                        color_params: Optional[Dict] = None) -> np.ndarray:
        """
        Convert grayscale texture to colored RGB.
        
        Args:
            texture: Grayscale texture [0, 1]
            color_map: 'grayscale', 'gradient', or 'custom'
            color_params: Parameters for colorization
            
        Returns:
            RGB texture (H x W x 3)
        """
        color_params = color_params or {}
        
        if color_map == 'grayscale':
            # Simple grayscale to RGB
            return np.stack([texture, texture, texture], axis=-1)
        
        elif color_map == 'gradient':
            # Linear gradient between two colors
            color1 = np.array(color_params.get('color1', [0.0, 0.0, 0.0]))
            color2 = np.array(color_params.get('color2', [1.0, 1.0, 1.0]))
            
            rgb = np.zeros((*texture.shape, 3), dtype=np.float32)
            for i in range(3):
                rgb[:, :, i] = color1[i] + texture * (color2[i] - color1[i])
            
            return rgb
        
        elif color_map == 'custom':
            # Custom color transfer
            # This would map texture values to specific colors
            # For now, default to grayscale
            return np.stack([texture, texture, texture], axis=-1)
        
        else:
            return np.stack([texture, texture, texture], axis=-1)


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Generate different textures
    shape = (256, 256)
    
    perlin = ProceduralTextures.perlin_noise_2d(shape, scale=20, seed=42)
    worley = ProceduralTextures.worley_noise_2d(shape, num_points=15, seed=42)
    fbm = ProceduralTextures.fbm_noise_2d(shape, base_scale=15, octaves=6, seed=42)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(perlin, cmap='gray')
    axes[0].set_title('Perlin Noise')
    axes[1].imshow(worley, cmap='gray')
    axes[1].set_title('Worley Noise')
    axes[2].imshow(fbm, cmap='gray')
    axes[2].set_title('fBM')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/tmp/procedural_textures.png', dpi=150, bbox_inches='tight')
    print("✅ Generated texture samples: /tmp/procedural_textures.png")
    
    # Test colorization
    colored = ProceduralTextures.colorize_texture(
        perlin,
        color_map='gradient',
        color_params={'color1': [0.1, 0.2, 0.5], 'color2': [0.9, 0.7, 0.3]}
    )
    print(f"✅ Colored texture shape: {colored.shape}")

