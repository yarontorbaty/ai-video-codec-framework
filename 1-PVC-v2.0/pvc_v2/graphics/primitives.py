#!/usr/bin/env python3
"""
PVC v2.0 Graphics Primitives Library

Core rendering functions for Neural-Procedural Hybrid Codec.
These functions are embedded in the decoder (zero transmission cost).
Only parameters are transmitted.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class FunctionCall:
    """Represents a single graphics function call."""
    func_id: int
    func_name: str
    params: dict
    normalized_params: dict = None  # NEW: Normalized parameters for training
    
    def estimate_size(self) -> int:
        """Estimate compressed size in bytes."""
        size = 2  # func_id (1 byte) + param_count (1 byte)
        
        for key, value in self.params.items():
            if isinstance(value, (int, float)):
                size += 2  # Quantized numeric
            elif isinstance(value, (list, tuple)):
                if len(value) == 3:  # RGB color
                    size += 3
                elif len(value) == 2:  # Coordinate pair
                    size += 4
                else:
                    size += len(value) * 2
        
        return size
    
    def get_normalized_params(self, frame_width: int = 256, frame_height: int = 256) -> np.ndarray:
        """
        Get parameters as normalized numpy array for neural network training.
        
        Returns:
            Array of shape (10,) with normalized parameters:
            [coords(4), color1(3), color2(3)]
        """
        if self.normalized_params is not None:
            # Use pre-computed normalized params
            coords = self.normalized_params.get('coords', [0, 0, 0, 0])
            color1 = self.normalized_params.get('color1', [0.5, 0.5, 0.5])
            color2 = self.normalized_params.get('color2', [0.5, 0.5, 0.5])
            return np.array(coords + color1 + color2, dtype=np.float32)
        
        # Compute from params (fallback)
        # Extract and normalize coordinates
        if 'x' in self.params:
            coords = [
                self.params.get('x', 0) / frame_width,
                self.params.get('y', 0) / frame_height,
                self.params.get('w', frame_width) / frame_width,
                self.params.get('h', frame_height) / frame_height,
            ]
        elif 'cx' in self.params:
            coords = [
                self.params.get('cx', frame_width/2) / frame_width,
                self.params.get('cy', frame_height/2) / frame_height,
                self.params.get('rx', 50) / frame_width,
                self.params.get('ry', 50) / frame_height,
            ]
        else:
            coords = [0.5, 0.5, 0.2, 0.2]
        
        # Extract colors (already normalized 0-1)
        color1 = self.params.get('color', self.params.get('fill', self.params.get('color_inner', [0.5, 0.5, 0.5])))
        if color1 is None:
            color1 = [0.5, 0.5, 0.5]
        elif isinstance(color1, (list, tuple)):
            color1 = list(color1)[:3]  # Ensure 3 elements
            while len(color1) < 3:
                color1.append(0.5)
        else:
            color1 = [0.5, 0.5, 0.5]
        
        color2 = self.params.get('color2', self.params.get('stroke', self.params.get('color_outer', [0.5, 0.5, 0.5])))
        if color2 is None:
            color2 = [0.5, 0.5, 0.5]
        elif isinstance(color2, (list, tuple)):
            color2 = list(color2)[:3]  # Ensure 3 elements
            while len(color2) < 3:
                color2.append(0.5)
        else:
            color2 = [0.5, 0.5, 0.5]
        
        return np.array(coords + color1 + color2, dtype=np.float32)


class GraphicsPrimitives:
    """
    Core graphics functions for procedural video reconstruction.
    
    These functions are built into the decoder and never transmitted.
    Only their parameters are sent, achieving extreme compression.
    """
    
    # Function ID mapping (for encoding)
    FUNCTION_IDS = {
        'fill_solid': 0,
        'draw_gradient_linear': 1,
        'draw_gradient_radial': 2,
        'draw_rectangle': 3,
        'draw_ellipse': 4,
        'draw_polygon': 5,
        'apply_gaussian_blur': 6,
        'apply_noise': 7,
        'blend_layers': 8,
        'adjust_brightness': 9,
    }
    
    def __init__(self, width: int = 1920, height: int = 1080):
        """Initialize renderer."""
        self.width = width
        self.height = height
        self.canvas = None
        self.layers = []
    
    def create_canvas(self, background: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        """Create a blank canvas."""
        self.canvas = np.ones((self.height, self.width, 3), dtype=np.float32)
        self.canvas[:, :] = background
        return self.canvas
    
    def fill_solid(self, 
                   x: int, y: int, w: int, h: int,
                   color: Tuple[float, float, float],
                   opacity: float = 1.0) -> FunctionCall:
        """
        Fill a rectangular region with solid color.
        
        Args:
            x, y: Top-left corner
            w, h: Width and height
            color: RGB color (0-1 range)
            opacity: Opacity (0-1)
        """
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.width, x + w), min(self.height, y + h)
        
        if x2 > x1 and y2 > y1:
            color_arr = np.array(color, dtype=np.float32)
            self.canvas[y1:y2, x1:x2] = (
                self.canvas[y1:y2, x1:x2] * (1 - opacity) +
                color_arr * opacity
            )
        
        return FunctionCall(
            func_id=0,
            func_name='fill_solid',
            params={'x': x, 'y': y, 'w': w, 'h': h, 'color': color, 'opacity': opacity}
        )
    
    def draw_gradient_linear(self,
                            x: int, y: int, w: int, h: int,
                            color1: Tuple[float, float, float],
                            color2: Tuple[float, float, float],
                            angle: float = 0.0) -> FunctionCall:
        """
        Draw a linear gradient.
        
        Args:
            x, y: Top-left corner
            w, h: Width and height
            color1: Start color (RGB, 0-1)
            color2: End color (RGB, 0-1)
            angle: Angle in degrees (0=horizontal, 90=vertical)
        """
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.width, x + w), min(self.height, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return FunctionCall(1, 'draw_gradient_linear', 
                              {'x': x, 'y': y, 'w': w, 'h': h, 
                               'color1': color1, 'color2': color2, 'angle': angle})
        
        # Create gradient
        region_h, region_w = y2 - y1, x2 - x1
        
        # Compute gradient direction
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Create coordinate grids
        y_grid, x_grid = np.mgrid[0:region_h, 0:region_w]
        
        # Project coordinates onto gradient axis
        projection = (x_grid * cos_a + y_grid * sin_a)
        projection = projection - projection.min()
        if projection.max() > 0:
            projection = projection / projection.max()
        
        # Interpolate colors
        c1 = np.array(color1, dtype=np.float32)
        c2 = np.array(color2, dtype=np.float32)
        
        gradient = c1 + (c2 - c1) * projection[:, :, np.newaxis]
        
        # Apply to canvas
        self.canvas[y1:y2, x1:x2] = gradient
        
        return FunctionCall(
            func_id=1,
            func_name='draw_gradient_linear',
            params={'x': x, 'y': y, 'w': w, 'h': h, 
                   'color1': color1, 'color2': color2, 'angle': angle}
        )
    
    def draw_gradient_radial(self,
                            cx: int, cy: int, radius: int,
                            color_inner: Tuple[float, float, float],
                            color_outer: Tuple[float, float, float]) -> FunctionCall:
        """
        Draw a radial gradient.
        
        Args:
            cx, cy: Center coordinates
            radius: Gradient radius
            color_inner: Inner color (RGB, 0-1)
            color_outer: Outer color (RGB, 0-1)
        """
        # Create distance map
        y_grid, x_grid = np.mgrid[0:self.height, 0:self.width]
        dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        
        # Normalize distance
        dist = np.clip(dist / radius, 0, 1)
        
        # Interpolate colors
        c_inner = np.array(color_inner, dtype=np.float32)
        c_outer = np.array(color_outer, dtype=np.float32)
        
        gradient = c_inner + (c_outer - c_inner) * dist[:, :, np.newaxis]
        
        # Apply to canvas
        self.canvas[:, :] = gradient
        
        return FunctionCall(
            func_id=2,
            func_name='draw_gradient_radial',
            params={'cx': cx, 'cy': cy, 'radius': radius,
                   'color_inner': color_inner, 'color_outer': color_outer}
        )
    
    def draw_rectangle(self,
                      x: int, y: int, w: int, h: int,
                      fill: Optional[Tuple[float, float, float]] = None,
                      stroke: Optional[Tuple[float, float, float]] = None,
                      stroke_width: int = 1,
                      opacity: float = 1.0) -> FunctionCall:
        """
        Draw a rectangle.
        
        Args:
            x, y: Top-left corner
            w, h: Width and height
            fill: Fill color (RGB, 0-1) or None
            stroke: Stroke color (RGB, 0-1) or None
            stroke_width: Stroke width in pixels
            opacity: Opacity (0-1)
        """
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.width, x + w), min(self.height, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return FunctionCall(3, 'draw_rectangle',
                              {'x': x, 'y': y, 'w': w, 'h': h,
                               'fill': fill, 'stroke': stroke,
                               'stroke_width': stroke_width, 'opacity': opacity})
        
        # Draw fill
        if fill is not None:
            fill_arr = np.array(fill, dtype=np.float32)
            self.canvas[y1:y2, x1:x2] = (
                self.canvas[y1:y2, x1:x2] * (1 - opacity) +
                fill_arr * opacity
            )
        
        # Draw stroke
        if stroke is not None and stroke_width > 0:
            stroke_arr = np.array(stroke, dtype=np.float32)
            # Top edge
            self.canvas[y1:y1+stroke_width, x1:x2] = stroke_arr
            # Bottom edge
            self.canvas[y2-stroke_width:y2, x1:x2] = stroke_arr
            # Left edge
            self.canvas[y1:y2, x1:x1+stroke_width] = stroke_arr
            # Right edge
            self.canvas[y1:y2, x2-stroke_width:x2] = stroke_arr
        
        return FunctionCall(
            func_id=3,
            func_name='draw_rectangle',
            params={'x': x, 'y': y, 'w': w, 'h': h,
                   'fill': fill, 'stroke': stroke,
                   'stroke_width': stroke_width, 'opacity': opacity}
        )
    
    def draw_ellipse(self,
                    cx: int, cy: int, rx: int, ry: int,
                    fill: Optional[Tuple[float, float, float]] = None,
                    stroke: Optional[Tuple[float, float, float]] = None,
                    stroke_width: int = 1,
                    opacity: float = 1.0) -> FunctionCall:
        """
        Draw an ellipse.
        
        Args:
            cx, cy: Center coordinates
            rx, ry: X and Y radii
            fill: Fill color (RGB, 0-1) or None
            stroke: Stroke color (RGB, 0-1) or None
            stroke_width: Stroke width in pixels
            opacity: Opacity (0-1)
        """
        # Create mask for ellipse
        y_grid, x_grid = np.mgrid[0:self.height, 0:self.width]
        dist = ((x_grid - cx) / max(rx, 1))**2 + ((y_grid - cy) / max(ry, 1))**2
        
        # Fill
        if fill is not None:
            mask = (dist <= 1).astype(np.float32)
            fill_arr = np.array(fill, dtype=np.float32)
            for c in range(3):
                self.canvas[:, :, c] = (
                    self.canvas[:, :, c] * (1 - mask * opacity) +
                    fill_arr[c] * mask * opacity
                )
        
        # Stroke
        if stroke is not None and stroke_width > 0:
            inner_r = max(0, 1 - stroke_width / max(rx, ry))
            stroke_mask = ((dist <= 1) & (dist >= inner_r**2)).astype(np.float32)
            stroke_arr = np.array(stroke, dtype=np.float32)
            for c in range(3):
                self.canvas[:, :, c] = (
                    self.canvas[:, :, c] * (1 - stroke_mask * opacity) +
                    stroke_arr[c] * stroke_mask * opacity
                )
        
        return FunctionCall(
            func_id=4,
            func_name='draw_ellipse',
            params={'cx': cx, 'cy': cy, 'rx': rx, 'ry': ry,
                   'fill': fill, 'stroke': stroke,
                   'stroke_width': stroke_width, 'opacity': opacity}
        )
    
    def draw_polygon(self,
                    points: List[Tuple[int, int]],
                    fill: Optional[Tuple[float, float, float]] = None,
                    stroke: Optional[Tuple[float, float, float]] = None,
                    stroke_width: int = 1,
                    opacity: float = 1.0) -> FunctionCall:
        """
        Draw a polygon.
        
        Args:
            points: List of (x, y) coordinates
            fill: Fill color (RGB, 0-1) or None
            stroke: Stroke color (RGB, 0-1) or None
            stroke_width: Stroke width in pixels
            opacity: Opacity (0-1)
        """
        if len(points) < 3:
            return FunctionCall(5, 'draw_polygon',
                              {'points': points, 'fill': fill, 'stroke': stroke,
                               'stroke_width': stroke_width, 'opacity': opacity})
        
        # Convert to uint8 for OpenCV
        canvas_uint8 = (self.canvas * 255).astype(np.uint8)
        
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        
        # Fill
        if fill is not None:
            fill_bgr = tuple(int(c * 255) for c in reversed(fill))
            cv2.fillPoly(canvas_uint8, [pts], fill_bgr)
        
        # Stroke
        if stroke is not None and stroke_width > 0:
            stroke_bgr = tuple(int(c * 255) for c in reversed(stroke))
            cv2.polylines(canvas_uint8, [pts], True, stroke_bgr, stroke_width)
        
        # Convert back to float
        self.canvas = canvas_uint8.astype(np.float32) / 255.0
        
        return FunctionCall(
            func_id=5,
            func_name='draw_polygon',
            params={'points': points, 'fill': fill, 'stroke': stroke,
                   'stroke_width': stroke_width, 'opacity': opacity}
        )
    
    def apply_gaussian_blur(self,
                           x: int, y: int, w: int, h: int,
                           radius: int) -> FunctionCall:
        """
        Apply Gaussian blur to a region.
        
        Args:
            x, y: Top-left corner of region
            w, h: Region size
            radius: Blur radius (kernel size = 2*radius+1)
        """
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.width, x + w), min(self.height, y + h)
        
        if x2 > x1 and y2 > y1 and radius > 0:
            region = self.canvas[y1:y2, x1:x2]
            kernel_size = 2 * radius + 1
            blurred = cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)
            self.canvas[y1:y2, x1:x2] = blurred
        
        return FunctionCall(
            func_id=6,
            func_name='apply_gaussian_blur',
            params={'x': x, 'y': y, 'w': w, 'h': h, 'radius': radius}
        )
    
    def apply_noise(self,
                   x: int, y: int, w: int, h: int,
                   strength: float = 0.1,
                   seed: int = 0) -> FunctionCall:
        """
        Apply random noise to a region.
        
        Args:
            x, y: Top-left corner of region
            w, h: Region size
            strength: Noise strength (0-1)
            seed: Random seed for reproducibility
        """
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.width, x + w), min(self.height, y + h)
        
        if x2 > x1 and y2 > y1:
            np.random.seed(seed)
            noise = np.random.randn(y2 - y1, x2 - x1, 3) * strength
            self.canvas[y1:y2, x1:x2] = np.clip(
                self.canvas[y1:y2, x1:x2] + noise, 0, 1
            )
        
        return FunctionCall(
            func_id=7,
            func_name='apply_noise',
            params={'x': x, 'y': y, 'w': w, 'h': h, 'strength': strength, 'seed': seed}
        )
    
    def adjust_brightness(self,
                         x: int, y: int, w: int, h: int,
                         factor: float) -> FunctionCall:
        """
        Adjust brightness of a region.
        
        Args:
            x, y: Top-left corner of region
            w, h: Region size
            factor: Brightness factor (1.0 = no change, >1 = brighter, <1 = darker)
        """
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.width, x + w), min(self.height, y + h)
        
        if x2 > x1 and y2 > y1:
            self.canvas[y1:y2, x1:x2] = np.clip(
                self.canvas[y1:y2, x1:x2] * factor, 0, 1
            )
        
        return FunctionCall(
            func_id=9,
            func_name='adjust_brightness',
            params={'x': x, 'y': y, 'w': w, 'h': h, 'factor': factor}
        )
    
    def get_canvas(self) -> np.ndarray:
        """Get the current canvas as uint8 image."""
        return (self.canvas * 255).astype(np.uint8)
    
    def execute_sequence(self, function_calls: List[FunctionCall]) -> np.ndarray:
        """
        Execute a sequence of function calls.
        
        Args:
            function_calls: List of FunctionCall objects
            
        Returns:
            Rendered frame
        """
        self.create_canvas()
        
        for call in function_calls:
            # Execute the function by name
            func = getattr(self, call.func_name)
            func(**call.params)
        
        return self.get_canvas()

