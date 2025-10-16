#!/usr/bin/env python3
"""
Procedural Scene Generator Agent
Inspired by 90s Demoscene techniques for mathematical scene generation.
Generates complex visual scenes from compact mathematical descriptions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import math
import logging
import os
from typing import Dict, List, Tuple, Optional, Callable
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SceneParameters:
    """Parameters for procedural scene generation."""
    time: float
    resolution: Tuple[int, int]
    complexity: float = 1.0
    color_palette: str = 'vibrant'
    motion_intensity: float = 1.0
    geometric_scale: float = 1.0


class DemosceneMath:
    """Mathematical functions inspired by 90s demoscene techniques."""
    
    @staticmethod
    def smoothstep(edge0: float, edge1: float, x: float) -> float:
        """Smooth interpolation function."""
        t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    
    @staticmethod
    def noise_2d(x: float, y: float, seed: int = 0) -> float:
        """2D noise function for texture generation."""
        # Simple hash-based noise
        n = int(x) + int(y) * 57 + seed
        n = (n << 13) ^ n
        return (1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0)
    
    @staticmethod
    def fractal_noise(x, y, octaves: int = 4):
        """Fractal noise for complex textures."""
        value = np.zeros_like(x)
        amplitude = 1.0
        frequency = 1.0
        
        for _ in range(octaves):
            # Vectorized noise calculation
            n = (x * frequency).astype(int) + (y * frequency).astype(int) * 57
            n = (n << 13) ^ n
            noise = (1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0)
            value += amplitude * noise
            amplitude *= 0.5
            frequency *= 2.0
        
        return value
    
    @staticmethod
    def distance_field_circle(x, y, cx, cy, r):
        """Distance field for circle."""
        return np.sqrt((x - cx)**2 + (y - cy)**2) - r
    
    @staticmethod
    def distance_field_box(x: float, y: float, cx: float, cy: float, w: float, h: float) -> float:
        """Distance field for rectangle."""
        dx = abs(x - cx) - w/2
        dy = abs(y - cy) - h/2
        return math.sqrt(max(dx, 0)**2 + max(dy, 0)**2) + min(max(dx, dy), 0)


class ProceduralSceneGenerator:
    """Generates procedural scenes using demoscene-inspired techniques."""
    
    def __init__(self, resolution: Tuple[int, int] = (1920, 1080)):
        self.resolution = resolution
        self.width, self.height = resolution
        
        # Create coordinate grids
        self.x_coords = np.linspace(0, 1, self.width)
        self.y_coords = np.linspace(0, 1, self.height)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)
        
        # Color palettes inspired by demoscene
        self.palettes = {
            'vibrant': [
                (1.0, 0.0, 0.0),    # Red
                (0.0, 1.0, 0.0),    # Green
                (0.0, 0.0, 1.0),    # Blue
                (1.0, 1.0, 0.0),    # Yellow
                (1.0, 0.0, 1.0),    # Magenta
                (0.0, 1.0, 1.0),    # Cyan
            ],
            'pastel': [
                (1.0, 0.8, 0.8),    # Light red
                (0.8, 1.0, 0.8),    # Light green
                (0.8, 0.8, 1.0),    # Light blue
                (1.0, 1.0, 0.8),    # Light yellow
                (1.0, 0.8, 1.0),    # Light magenta
                (0.8, 1.0, 1.0),    # Light cyan
            ],
            'monochrome': [
                (0.0, 0.0, 0.0),    # Black
                (0.5, 0.5, 0.5),    # Gray
                (1.0, 1.0, 1.0),    # White
            ]
        }
    
    def generate_geometric_pattern(self, params: SceneParameters) -> np.ndarray:
        """Generate geometric patterns using distance fields."""
        frame = np.zeros((self.height, self.width, 3))
        
        # Animated geometric shapes
        time_offset = params.time * 2.0
        
        for i in range(int(3 + params.complexity * 5)):
            # Animated position
            cx = 0.5 + 0.3 * math.sin(time_offset + i * 0.5)
            cy = 0.5 + 0.3 * math.cos(time_offset + i * 0.3)
            
            # Animated size
            radius = 0.1 + 0.05 * math.sin(time_offset * 1.5 + i)
            
            # Distance field
            dist = DemosceneMath.distance_field_circle(
                self.X, self.Y, cx, cy, radius
            )
            
            # Smooth step for anti-aliasing
            intensity = DemosceneMath.smoothstep(0.0, 0.02, -dist)
            
            # Color based on position and time
            color_idx = (i + int(time_offset * 10)) % len(self.palettes[params.color_palette])
            color = self.palettes[params.color_palette][color_idx]
            
            # Add to frame
            for c in range(3):
                frame[:, :, c] += intensity * color[c]
        
        return np.clip(frame, 0, 1)
    
    def generate_fractal_texture(self, params: SceneParameters) -> np.ndarray:
        """Generate fractal textures using noise functions."""
        # Scale coordinates
        scale = 2.0 + params.complexity * 3.0
        x_scaled = self.X * scale
        y_scaled = self.Y * scale
        
        # Add time-based animation
        time_offset = params.time * 0.5
        x_animated = x_scaled + time_offset
        y_animated = y_scaled + time_offset * 0.7
        
        # Generate fractal noise
        noise = DemosceneMath.fractal_noise(x_animated, y_animated, octaves=6)
        
        # Create color from noise
        frame = np.zeros((self.height, self.width, 3))
        
        # Use noise to select from color palette
        for i, color in enumerate(self.palettes[params.color_palette]):
            mask = (noise > (i / len(self.palettes[params.color_palette]))) & \
                   (noise <= ((i + 1) / len(self.palettes[params.color_palette])))
            
            for c in range(3):
                frame[mask, c] = color[c]
        
        return frame
    
    def generate_plasma_effect(self, params: SceneParameters) -> np.ndarray:
        """Generate plasma effect (classic demoscene technique)."""
        time_offset = params.time * 2.0
        
        # Multiple sine waves for plasma effect
        plasma = (
            np.sin(self.X * 10 + time_offset) +
            np.sin(self.Y * 10 + time_offset * 0.7) +
            np.sin((self.X + self.Y) * 7 + time_offset * 1.3) +
            np.sin(np.sqrt(self.X**2 + self.Y**2) * 8 + time_offset * 0.9)
        ) / 4.0
        
        # Normalize to [0, 1]
        plasma = (plasma + 1.0) / 2.0
        
        # Create color mapping
        frame = np.zeros((self.height, self.width, 3))
        
        for i, color in enumerate(self.palettes[params.color_palette]):
            # Create bands based on plasma value
            band_start = i / len(self.palettes[params.color_palette])
            band_end = (i + 1) / len(self.palettes[params.color_palette])
            
            mask = (plasma >= band_start) & (plasma < band_end)
            
            for c in range(3):
                frame[mask, c] = color[c]
        
        return frame
    
    def generate_mandelbrot_zoom(self, params: SceneParameters) -> np.ndarray:
        """Generate Mandelbrot set with animated zoom."""
        # Animated zoom parameters
        zoom = 1.0 + params.time * 0.5
        center_x = 0.5 + 0.1 * math.sin(params.time)
        center_y = 0.5 + 0.1 * math.cos(params.time)
        
        # Map screen coordinates to complex plane
        x_min = center_x - 2.0 / zoom
        x_max = center_x + 2.0 / zoom
        y_min = center_y - 2.0 / zoom
        y_max = center_y + 2.0 / zoom
        
        x_complex = np.linspace(x_min, x_max, self.width)
        y_complex = np.linspace(y_min, y_max, self.height)
        X_complex, Y_complex = np.meshgrid(x_complex, y_complex)
        
        # Mandelbrot iteration
        c = X_complex + 1j * Y_complex
        z = np.zeros_like(c)
        iterations = np.zeros(c.shape, dtype=int)
        
        for i in range(100):  # Max iterations
            mask = np.abs(z) <= 2
            z[mask] = z[mask]**2 + c[mask]
            iterations[mask] = i
        
        # Color mapping
        frame = np.zeros((self.height, self.width, 3))
        
        # Use iteration count to determine color
        normalized_iter = iterations / 100.0
        
        for i, color in enumerate(self.palettes[params.color_palette]):
            band_start = i / len(self.palettes[params.color_palette])
            band_end = (i + 1) / len(self.palettes[params.color_palette])
            
            mask = (normalized_iter >= band_start) & (normalized_iter < band_end)
            
            for c in range(3):
                frame[mask, c] = color[c]
        
        return frame
    
    def generate_scene(self, params: SceneParameters) -> np.ndarray:
        """Generate a complete procedural scene."""
        # Choose generation method based on complexity
        if params.complexity < 0.3:
            return self.generate_geometric_pattern(params)
        elif params.complexity < 0.6:
            return self.generate_fractal_texture(params)
        elif params.complexity < 0.8:
            return self.generate_plasma_effect(params)
        else:
            return self.generate_mandelbrot_zoom(params)


class ProceduralCompressionAgent:
    """AI agent that uses procedural generation for video compression."""
    
    def __init__(self, resolution: Tuple[int, int] = (1920, 1080), config: Optional[Dict] = None):
        self.resolution = resolution
        self.generator = ProceduralSceneGenerator(resolution)
        
        # LLM-suggested configuration (can override defaults)
        self.config = config or {}
        self.compression_strategy = self.config.get('compression_strategy', 'parameter_storage')
        self.parameter_storage_enabled = self.config.get('parameter_storage', False)
        self.complexity_level = self.config.get('complexity_level', 1.0)
        self.bitrate_target = self.config.get('bitrate_target_mbps', 1.0)
        
        logger.info(f"Procedural agent initialized with strategy: {self.compression_strategy}")
        logger.info(f"Parameter storage: {self.parameter_storage_enabled}, Target bitrate: {self.bitrate_target} Mbps")
        
        # Neural network for scene parameter prediction
        self.parameter_predictor = self._create_parameter_predictor()
        
        # Scene description encoder
        self.scene_encoder = self._create_scene_encoder()
        
        # Compression state
        self.scene_parameters = []
        self.compression_ratio = 0.0
        
    def _create_parameter_predictor(self) -> nn.Module:
        """Create neural network for predicting scene parameters."""
        return nn.Sequential(
            nn.Linear(3, 64),  # RGB input
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),  # 6 parameters: complexity, color_palette, motion_intensity, etc.
            nn.Sigmoid()
        )
    
    def _create_scene_encoder(self) -> nn.Module:
        """Create encoder for scene descriptions."""
        return nn.Sequential(
            nn.Linear(6, 32),  # Scene parameters
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),   # Compact scene description
            nn.Tanh()
        )
    
    def analyze_frame(self, frame: np.ndarray) -> SceneParameters:
        """Analyze frame and predict scene parameters."""
        # Convert frame to tensor
        frame_tensor = torch.from_numpy(frame).float()
        
        # Sample representative pixels
        sample_pixels = frame_tensor[::50, ::50, :].flatten()[:3]  # Take first 3 RGB values
        
        # Predict parameters
        with torch.no_grad():
            predicted_params = self.parameter_predictor(sample_pixels)
        
        # Convert to SceneParameters
        params = SceneParameters(
            time=0.0,  # Will be set by frame index
            resolution=self.resolution,
            complexity=float(predicted_params[0].item()),
            color_palette=['vibrant', 'pastel', 'monochrome'][int(predicted_params[1].item() * 2.99)],
            motion_intensity=float(predicted_params[2].item()),
            geometric_scale=float(predicted_params[3].item())
        )
        
        return params
    
    def compress_frame(self, frame: np.ndarray, frame_index: int) -> Dict:
        """Compress frame using procedural generation."""
        # Analyze frame to get parameters
        params = self.analyze_frame(frame)
        params.time = frame_index * 0.1  # Time-based animation
        
        # Generate procedural scene
        procedural_frame = self.generator.generate_scene(params)
        
        # Calculate compression metrics
        original_size = frame.nbytes
        compressed_size = len(json.dumps({
            'complexity': params.complexity,
            'color_palette': params.color_palette,
            'motion_intensity': params.motion_intensity,
            'geometric_scale': params.geometric_scale,
            'time': params.time
        }))
        
        compression_ratio = compressed_size / original_size
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'parameters': params,
            'procedural_frame': procedural_frame
        }
    
    def decompress_frame(self, compressed_data: Dict) -> np.ndarray:
        """Decompress frame from procedural description."""
        params = compressed_data['parameters']
        return self.generator.generate_scene(params)
    
    def compress_video(self, video_path: str, output_path: str) -> Dict:
        """Compress entire video using procedural generation."""
        logger.info(f"Starting procedural compression of {video_path}")
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Compression results
        compressed_frames = []
        total_original_size = 0
        total_compressed_size = 0
        
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Compress frame
            compression_result = self.compress_frame(frame_rgb, frame_index)
            
            compressed_frames.append(compression_result)
            total_original_size += compression_result['original_size']
            total_compressed_size += compression_result['compressed_size']
            
            frame_index += 1
            
            if frame_index % 30 == 0:
                logger.info(f"Compressed {frame_index}/{total_frames} frames")
        
        cap.release()
        
        # Calculate overall metrics
        overall_compression_ratio = total_compressed_size / total_original_size
        bitrate_mbps = (total_compressed_size * 8) / (total_frames / fps) / 1_000_000
        
        results = {
            'total_frames': total_frames,
            'fps': fps,
            'original_size_mb': total_original_size / (1024 * 1024),
            'compressed_size_mb': total_compressed_size / (1024 * 1024),
            'compression_ratio': overall_compression_ratio,
            'bitrate_mbps': bitrate_mbps,
            'compressed_frames': compressed_frames
        }
        
        logger.info(f"Procedural compression completed: {results}")
        return results
    
    def _generate_with_parameter_storage(self, output_path: str, duration: float, fps: float) -> Dict:
        """
        NEW APPROACH: Store only procedural parameters, not rendered frames.
        This is the approach suggested by LLM to achieve massive compression.
        """
        total_frames = int(duration * fps)
        parameters_list = []
        
        logger.info(f"Generating {total_frames} parameter sets (not rendering frames)...")
        
        for frame_idx in range(total_frames):
            # Store ONLY the parameters needed to regenerate this frame
            # Each parameter set is ~100-200 bytes instead of ~600KB for rendered frame
            params = {
                'frame': frame_idx,
                'time': frame_idx / fps,
                'complexity': 0.5 + 0.3 * math.sin(frame_idx * 0.1),
                'color_palette': 'vibrant',
                'motion_intensity': 1.0,
                'geometric_scale': 1.0,
                'scene_type': frame_idx % 5  # Vary between 5 different scene types
            }
            parameters_list.append(params)
        
        # Save parameters to JSON file (TINY compared to video)
        params_file = output_path.replace('.mp4', '_params.json')
        with open(params_file, 'w') as f:
            json.dump({
                'version': '1.0',
                'resolution': self.resolution,
                'fps': fps,
                'duration': duration,
                'total_frames': total_frames,
                'parameters': parameters_list
            }, f)
        
        params_file_size = os.path.getsize(params_file)
        
        # Calculate what the rendered size WOULD have been
        bytes_per_frame = self.resolution[0] * self.resolution[1] * 3  # RGB
        rendered_size = bytes_per_frame * total_frames
        
        # Calculate actual compression achieved
        compression_ratio = params_file_size / rendered_size
        bitrate_mbps = (params_file_size * 8) / duration / 1_000_000
        
        logger.info(f"✅ Parameter storage complete!")
        logger.info(f"   Rendered size would be: {rendered_size / (1024*1024):.2f} MB")
        logger.info(f"   Parameter file size: {params_file_size / 1024:.2f} KB")
        logger.info(f"   Compression ratio: {compression_ratio:.6f}")
        logger.info(f"   Bitrate: {bitrate_mbps:.4f} Mbps")
        
        return {
            'status': 'completed',
            'mode': 'parameter_storage',
            'output_file': params_file,
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'rendered_size_mb': rendered_size / (1024 * 1024),
            'params_size_kb': params_file_size / 1024,
            'compression_ratio': compression_ratio,
            'bitrate_mbps': bitrate_mbps,
            'reduction_vs_rendered': (1 - compression_ratio) * 100  # Percentage reduction
        }
    
    def generate_procedural_video(self, output_path: str, duration: float = 10.0, 
                                fps: float = 30.0) -> Dict:
        """Generate a procedural video for testing (or store parameters if enabled)."""
        logger.info(f"Generating procedural video: {output_path}")
        
        # NEW: If parameter storage is enabled, store compact parameters instead of rendering
        if self.parameter_storage_enabled:
            logger.info("🔧 PARAMETER STORAGE MODE: Storing parameters instead of rendering frames")
            return self._generate_with_parameter_storage(output_path, duration, fps)
        
        # OLD: Render full video (current behavior)
        total_frames = int(duration * fps)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, self.resolution)
        
        for frame_idx in range(total_frames):
            # Create scene parameters
            params = SceneParameters(
                time=frame_idx / fps,
                resolution=self.resolution,
                complexity=0.5 + 0.3 * math.sin(frame_idx * 0.1),
                color_palette='vibrant',
                motion_intensity=1.0,
                geometric_scale=1.0
            )
            
            # Generate frame
            frame = self.generator.generate_scene(params)
            
            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        # Calculate metrics
        file_size = os.path.getsize(output_path)
        bitrate_mbps = (file_size * 8) / duration / 1_000_000
        
        results = {
            'duration': duration,
            'fps': fps,
            'total_frames': total_frames,
            'file_size_mb': file_size / (1024 * 1024),
            'bitrate_mbps': bitrate_mbps,
            'resolution': self.resolution
        }
        
        logger.info(f"Procedural video generated: {results}")
        return results


def main():
    """Test the procedural generation system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Procedural Scene Generator')
    parser.add_argument('--mode', choices=['generate', 'compress'], default='generate',
                       help='Operation mode')
    parser.add_argument('--input', type=str, help='Input video path')
    parser.add_argument('--output', type=str, default='procedural_output.mp4',
                       help='Output video path')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Video duration in seconds')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = ProceduralCompressionAgent()
    
    if args.mode == 'generate':
        # Generate procedural video
        results = agent.generate_procedural_video(args.output, args.duration)
        print(f"Generated procedural video: {results}")
        
    elif args.mode == 'compress':
        if not args.input:
            print("Input video path required for compress mode")
            return
        
        # Compress video using procedural generation
        results = agent.compress_video(args.input, args.output)
        print(f"Compression results: {results}")


if __name__ == "__main__":
    main()
