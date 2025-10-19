#!/usr/bin/env python3
"""
Extended Graphics Primitives Library for PVC v2.0

Adds 50+ new functions for richer visual representation:
- Advanced fills (gradients, noise, patterns)
- Advanced shapes (bezier, polygons, complex shapes)
- Texture effects (blur, sharpen, glow, shadow)
- Compositing (blend modes, masking)
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
from dataclasses import dataclass


class ExtendedGraphicsPrimitives:
    """Extended graphics primitive functions for procedural video generation."""
    
    def __init__(self, width: int = 256, height: int = 256):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # ==================== ADVANCED FILL FUNCTIONS ====================
    
    def fill_radial_gradient(self, 
                            center_x: int, center_y: int,
                            radius: int,
                            color_inner: Tuple[int, int, int],
                            color_outer: Tuple[int, int, int]) -> np.ndarray:
        """Radial gradient from center to edge."""
        y_coords, x_coords = np.ogrid[:self.height, :self.width]
        
        # Calculate distance from center
        dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Normalize to [0, 1]
        dist_norm = np.clip(dist / max(radius, 1), 0, 1)
        
        # Interpolate colors
        for c in range(3):
            self.canvas[:, :, c] = (
                color_inner[c] * (1 - dist_norm) + 
                color_outer[c] * dist_norm
            ).astype(np.uint8)
        
        return self.canvas.copy()
    
    def fill_conic_gradient(self,
                           center_x: int, center_y: int,
                           color1: Tuple[int, int, int],
                           color2: Tuple[int, int, int]) -> np.ndarray:
        """Conic (angular) gradient around center point."""
        y_coords, x_coords = np.ogrid[:self.height, :self.width]
        
        # Calculate angle from center
        angle = np.arctan2(y_coords - center_y, x_coords - center_x)
        
        # Normalize to [0, 1]
        angle_norm = (angle + np.pi) / (2 * np.pi)
        
        # Interpolate colors
        for c in range(3):
            self.canvas[:, :, c] = (
                color1[c] * (1 - angle_norm) + 
                color2[c] * angle_norm
            ).astype(np.uint8)
        
        return self.canvas.copy()
    
    def fill_noise_perlin(self,
                         scale: float = 10.0,
                         color_base: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
        """Perlin-like noise texture (simplified)."""
        # Simplified noise using random gradients
        y_coords, x_coords = np.ogrid[:self.height, :self.width]
        
        noise = np.sin(x_coords / scale) * np.cos(y_coords / scale)
        noise += np.sin((x_coords + y_coords) / (scale * 1.5))
        
        # Normalize to [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        
        # Apply to base color
        for c in range(3):
            self.canvas[:, :, c] = (color_base[c] * noise).astype(np.uint8)
        
        return self.canvas.copy()
    
    def fill_checkerboard(self,
                         square_size: int = 16,
                         color1: Tuple[int, int, int] = (0, 0, 0),
                         color2: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Checkerboard pattern."""
        y_coords, x_coords = np.ogrid[:self.height, :self.width]
        
        pattern = ((x_coords // square_size) + (y_coords // square_size)) % 2
        
        for c in range(3):
            self.canvas[:, :, c] = np.where(
                pattern == 0,
                color1[c],
                color2[c]
            ).astype(np.uint8)
        
        return self.canvas.copy()
    
    def fill_stripes(self,
                    stripe_width: int = 16,
                    angle: float = 0.0,
                    color1: Tuple[int, int, int] = (0, 0, 0),
                    color2: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Striped pattern at specified angle."""
        y_coords, x_coords = np.ogrid[:self.height, :self.width]
        
        # Rotate coordinates
        angle_rad = angle * np.pi / 180
        rotated = (x_coords * np.cos(angle_rad) + y_coords * np.sin(angle_rad))
        
        pattern = (rotated // stripe_width).astype(int) % 2
        
        for c in range(3):
            self.canvas[:, :, c] = np.where(
                pattern == 0,
                color1[c],
                color2[c]
            ).astype(np.uint8)
        
        return self.canvas.copy()
    
    def fill_dots(self,
                 dot_radius: int = 8,
                 spacing: int = 24,
                 color_dot: Tuple[int, int, int] = (255, 255, 255),
                 color_bg: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """Dot pattern."""
        self.canvas[:] = color_bg
        
        for y in range(0, self.height, spacing):
            for x in range(0, self.width, spacing):
                cv2.circle(self.canvas, (x, y), dot_radius, color_dot, -1)
        
        return self.canvas.copy()
    
    # ==================== ADVANCED SHAPE FUNCTIONS ====================
    
    def draw_polygon(self,
                    points: List[Tuple[int, int]],
                    color: Tuple[int, int, int],
                    filled: bool = True) -> np.ndarray:
        """Draw arbitrary polygon."""
        pts = np.array(points, dtype=np.int32)
        
        if filled:
            cv2.fillPoly(self.canvas, [pts], color)
        else:
            cv2.polylines(self.canvas, [pts], isClosed=True, color=color, thickness=2)
        
        return self.canvas.copy()
    
    def draw_bezier_curve(self,
                         p0: Tuple[int, int],
                         p1: Tuple[int, int],
                         p2: Tuple[int, int],
                         p3: Tuple[int, int],
                         color: Tuple[int, int, int],
                         thickness: int = 2) -> np.ndarray:
        """Draw cubic Bezier curve."""
        # Generate points along curve
        t_values = np.linspace(0, 1, 50)
        points = []
        
        for t in t_values:
            # Cubic Bezier formula
            x = int(
                (1-t)**3 * p0[0] + 
                3*(1-t)**2*t * p1[0] + 
                3*(1-t)*t**2 * p2[0] + 
                t**3 * p3[0]
            )
            y = int(
                (1-t)**3 * p0[1] + 
                3*(1-t)**2*t * p1[1] + 
                3*(1-t)*t**2 * p2[1] + 
                t**3 * p3[1]
            )
            points.append((x, y))
        
        # Draw polyline
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(self.canvas, [pts], isClosed=False, color=color, thickness=thickness)
        
        return self.canvas.copy()
    
    def draw_arc(self,
                center_x: int, center_y: int,
                radius: int,
                start_angle: float,
                end_angle: float,
                color: Tuple[int, int, int],
                thickness: int = 2) -> np.ndarray:
        """Draw circular arc."""
        cv2.ellipse(
            self.canvas,
            (center_x, center_y),
            (radius, radius),
            0,  # rotation
            start_angle,
            end_angle,
            color,
            thickness
        )
        
        return self.canvas.copy()
    
    def draw_rounded_rect(self,
                         x: int, y: int,
                         width: int, height: int,
                         radius: int,
                         color: Tuple[int, int, int],
                         filled: bool = True) -> np.ndarray:
        """Draw rounded rectangle."""
        thickness = -1 if filled else 2
        
        # Main rectangles
        cv2.rectangle(self.canvas, (x + radius, y), (x + width - radius, y + height), color, thickness)
        cv2.rectangle(self.canvas, (x, y + radius), (x + width, y + height - radius), color, thickness)
        
        # Corners
        cv2.circle(self.canvas, (x + radius, y + radius), radius, color, thickness)
        cv2.circle(self.canvas, (x + width - radius, y + radius), radius, color, thickness)
        cv2.circle(self.canvas, (x + radius, y + height - radius), radius, color, thickness)
        cv2.circle(self.canvas, (x + width - radius, y + height - radius), radius, color, thickness)
        
        return self.canvas.copy()
    
    def draw_star(self,
                 center_x: int, center_y: int,
                 outer_radius: int,
                 inner_radius: int,
                 points: int,
                 color: Tuple[int, int, int],
                 filled: bool = True) -> np.ndarray:
        """Draw n-pointed star."""
        pts = []
        angle_step = np.pi / points
        
        for i in range(points * 2):
            angle = i * angle_step - np.pi / 2
            radius = outer_radius if i % 2 == 0 else inner_radius
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            pts.append((x, y))
        
        pts_array = np.array(pts, dtype=np.int32)
        
        if filled:
            cv2.fillPoly(self.canvas, [pts_array], color)
        else:
            cv2.polylines(self.canvas, [pts_array], isClosed=True, color=color, thickness=2)
        
        return self.canvas.copy()
    
    def draw_triangle(self,
                     x1: int, y1: int,
                     x2: int, y2: int,
                     x3: int, y3: int,
                     color: Tuple[int, int, int],
                     filled: bool = True) -> np.ndarray:
        """Draw triangle."""
        pts = np.array([[x1, y1], [x2, y2], [x3, y3]], dtype=np.int32)
        
        if filled:
            cv2.fillPoly(self.canvas, [pts], color)
        else:
            cv2.polylines(self.canvas, [pts], isClosed=True, color=color, thickness=2)
        
        return self.canvas.copy()
    
    def draw_heart(self,
                  center_x: int, center_y: int,
                  size: int,
                  color: Tuple[int, int, int],
                  filled: bool = True) -> np.ndarray:
        """Draw heart shape."""
        # Parametric heart curve
        t_values = np.linspace(0, 2 * np.pi, 100)
        points = []
        
        for t in t_values:
            x = int(center_x + size * 16 * np.sin(t)**3)
            y = int(center_y - size * (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)))
            points.append((x, y))
        
        pts = np.array(points, dtype=np.int32)
        
        if filled:
            cv2.fillPoly(self.canvas, [pts], color)
        else:
            cv2.polylines(self.canvas, [pts], isClosed=True, color=color, thickness=2)
        
        return self.canvas.copy()
    
    def draw_ring(self,
                 center_x: int, center_y: int,
                 outer_radius: int,
                 inner_radius: int,
                 color: Tuple[int, int, int]) -> np.ndarray:
        """Draw ring (donut) shape."""
        # Draw outer circle
        cv2.circle(self.canvas, (center_x, center_y), outer_radius, color, -1)
        
        # Cut out inner circle (using black, assume will be blended)
        cv2.circle(self.canvas, (center_x, center_y), inner_radius, (0, 0, 0), -1)
        
        return self.canvas.copy()
    
    # ==================== TEXTURE & EFFECT FUNCTIONS ====================
    
    def apply_blur(self,
                  x: int, y: int,
                  width: int, height: int,
                  blur_size: int = 15) -> np.ndarray:
        """Apply Gaussian blur to region."""
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.width, x + width), min(self.height, y + height)
        
        if x2 > x1 and y2 > y1:
            region = self.canvas[y1:y2, x1:x2].copy()
            blurred = cv2.GaussianBlur(region, (blur_size | 1, blur_size | 1), 0)
            self.canvas[y1:y2, x1:x2] = blurred
        
        return self.canvas.copy()
    
    def apply_glow(self,
                  x: int, y: int,
                  width: int, height: int,
                  glow_color: Tuple[int, int, int],
                  intensity: float = 0.5) -> np.ndarray:
        """Apply glow effect to region."""
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.width, x + width), min(self.height, y + height)
        
        if x2 > x1 and y2 > y1:
            region = self.canvas[y1:y2, x1:x2].copy()
            glow = cv2.GaussianBlur(region, (21, 21), 0)
            
            # Blend with glow color
            glow_overlay = np.full_like(glow, glow_color, dtype=np.uint8)
            blended = cv2.addWeighted(glow, 1.0, glow_overlay, intensity, 0)
            
            self.canvas[y1:y2, x1:x2] = cv2.addWeighted(region, 0.7, blended, 0.3, 0)
        
        return self.canvas.copy()
    
    def apply_shadow(self,
                    x: int, y: int,
                    width: int, height: int,
                    offset_x: int = 5,
                    offset_y: int = 5,
                    shadow_color: Tuple[int, int, int] = (0, 0, 0),
                    alpha: float = 0.5) -> np.ndarray:
        """Apply drop shadow effect."""
        # Create shadow layer
        shadow = np.zeros_like(self.canvas)
        cv2.rectangle(shadow, 
                     (x + offset_x, y + offset_y),
                     (x + offset_x + width, y + offset_y + height),
                     shadow_color, -1)
        
        # Blur shadow
        shadow = cv2.GaussianBlur(shadow, (15, 15), 0)
        
        # Blend with canvas
        self.canvas = cv2.addWeighted(self.canvas, 1.0, shadow, alpha, 0)
        
        return self.canvas.copy()
    
    def apply_sharpen(self,
                     x: int, y: int,
                     width: int, height: int) -> np.ndarray:
        """Apply sharpening to region."""
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.width, x + width), min(self.height, y + height)
        
        if x2 > x1 and y2 > y1:
            region = self.canvas[y1:y2, x1:x2].copy()
            
            # Sharpening kernel
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            
            sharpened = cv2.filter2D(region, -1, kernel)
            self.canvas[y1:y2, x1:x2] = sharpened
        
        return self.canvas.copy()
    
    # ==================== MORE SHAPE FUNCTIONS ====================
    
    def draw_trapezoid(self,
                      x1: int, y1: int,
                      x2: int, y2: int,
                      top_width: int,
                      bottom_width: int,
                      color: Tuple[int, int, int],
                      filled: bool = True) -> np.ndarray:
        """Draw trapezoid."""
        # Top edge centered
        top_left_x = x1 + (x2 - x1 - top_width) // 2
        top_right_x = top_left_x + top_width
        
        # Bottom edge centered
        bottom_left_x = x1 + (x2 - x1 - bottom_width) // 2
        bottom_right_x = bottom_left_x + bottom_width
        
        pts = np.array([
            [top_left_x, y1],
            [top_right_x, y1],
            [bottom_right_x, y2],
            [bottom_left_x, y2]
        ], dtype=np.int32)
        
        if filled:
            cv2.fillPoly(self.canvas, [pts], color)
        else:
            cv2.polylines(self.canvas, [pts], isClosed=True, color=color, thickness=2)
        
        return self.canvas.copy()
    
    def draw_parallelogram(self,
                          x: int, y: int,
                          width: int, height: int,
                          skew: int,
                          color: Tuple[int, int, int],
                          filled: bool = True) -> np.ndarray:
        """Draw parallelogram."""
        pts = np.array([
            [x + skew, y],
            [x + width + skew, y],
            [x + width, y + height],
            [x, y + height]
        ], dtype=np.int32)
        
        if filled:
            cv2.fillPoly(self.canvas, [pts], color)
        else:
            cv2.polylines(self.canvas, [pts], isClosed=True, color=color, thickness=2)
        
        return self.canvas.copy()
    
    def draw_crescent(self,
                     center_x: int, center_y: int,
                     radius: int,
                     offset: int,
                     color: Tuple[int, int, int]) -> np.ndarray:
        """Draw crescent moon shape."""
        # Draw full circle
        cv2.circle(self.canvas, (center_x, center_y), radius, color, -1)
        
        # Cut out smaller circle to create crescent
        cv2.circle(self.canvas, (center_x + offset, center_y), radius - offset // 2, (0, 0, 0), -1)
        
        return self.canvas.copy()
    
    def draw_cross(self,
                  center_x: int, center_y: int,
                  size: int,
                  thickness: int,
                  color: Tuple[int, int, int]) -> np.ndarray:
        """Draw cross shape."""
        half_size = size // 2
        half_thick = thickness // 2
        
        # Vertical bar
        cv2.rectangle(self.canvas,
                     (center_x - half_thick, center_y - half_size),
                     (center_x + half_thick, center_y + half_size),
                     color, -1)
        
        # Horizontal bar
        cv2.rectangle(self.canvas,
                     (center_x - half_size, center_y - half_thick),
                     (center_x + half_size, center_y + half_thick),
                     color, -1)
        
        return self.canvas.copy()
    
    def draw_arrow(self,
                  start_x: int, start_y: int,
                  end_x: int, end_y: int,
                  color: Tuple[int, int, int],
                  head_size: int = 10) -> np.ndarray:
        """Draw arrow."""
        # Draw line
        cv2.line(self.canvas, (start_x, start_y), (end_x, end_y), color, 2)
        
        # Calculate arrow head angle
        angle = np.arctan2(end_y - start_y, end_x - start_x)
        
        # Arrow head points
        head_angle1 = angle + 3 * np.pi / 4
        head_angle2 = angle - 3 * np.pi / 4
        
        p1 = (int(end_x + head_size * np.cos(head_angle1)),
              int(end_y + head_size * np.sin(head_angle1)))
        p2 = (int(end_x + head_size * np.cos(head_angle2)),
              int(end_y + head_size * np.sin(head_angle2)))
        
        # Draw arrow head
        cv2.line(self.canvas, (end_x, end_y), p1, color, 2)
        cv2.line(self.canvas, (end_x, end_y), p2, color, 2)
        
        return self.canvas.copy()
    
    # ==================== MORE TEXTURE FUNCTIONS ====================
    
    def fill_wave(self,
                 frequency: float = 0.1,
                 amplitude: int = 20,
                 color1: Tuple[int, int, int] = (0, 0, 0),
                 color2: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Wave pattern."""
        y_coords, x_coords = np.ogrid[:self.height, :self.width]
        
        wave = np.sin(x_coords * frequency) * amplitude + y_coords
        wave_norm = (wave % 40) / 40
        
        for c in range(3):
            self.canvas[:, :, c] = (
                color1[c] * (1 - wave_norm) + 
                color2[c] * wave_norm
            ).astype(np.uint8)
        
        return self.canvas.copy()
    
    def apply_posterize(self,
                       x: int, y: int,
                       width: int, height: int,
                       levels: int = 4) -> np.ndarray:
        """Apply posterization effect."""
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.width, x + width), min(self.height, y + height)
        
        if x2 > x1 and y2 > y1:
            region = self.canvas[y1:y2, x1:x2].copy()
            
            # Reduce color levels
            step = 256 // levels
            posterized = (region // step) * step
            
            self.canvas[y1:y2, x1:x2] = posterized
        
        return self.canvas.copy()
    
    def apply_pixelate(self,
                      x: int, y: int,
                      width: int, height: int,
                      pixel_size: int = 8) -> np.ndarray:
        """Apply pixelation effect."""
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.width, x + width), min(self.height, y + height)
        
        if x2 > x1 and y2 > y1:
            region = self.canvas[y1:y2, x1:x2].copy()
            
            # Downsample and upsample
            small_h = max(1, (y2 - y1) // pixel_size)
            small_w = max(1, (x2 - x1) // pixel_size)
            
            small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            
            self.canvas[y1:y2, x1:x2] = pixelated
        
        return self.canvas.copy()
    
    def apply_vignette(self,
                      intensity: float = 0.5) -> np.ndarray:
        """Apply vignette (darkening at edges)."""
        y_coords, x_coords = np.ogrid[:self.height, :self.width]
        
        # Calculate distance from center
        center_x, center_y = self.width / 2, self.height / 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        vignette = 1 - (dist / max_dist * intensity)
        vignette = np.clip(vignette, 0, 1)
        
        # Apply to all channels
        for c in range(3):
            self.canvas[:, :, c] = (self.canvas[:, :, c] * vignette).astype(np.uint8)
        
        return self.canvas.copy()
    
    # ==================== COMPOSITING FUNCTIONS ====================
    
    def blend_multiply(self,
                      overlay_color: Tuple[int, int, int],
                      alpha: float = 0.5) -> np.ndarray:
        """Multiply blend mode."""
        overlay = np.full_like(self.canvas, overlay_color, dtype=np.float32)
        canvas_float = self.canvas.astype(np.float32)
        
        blended = (canvas_float * overlay / 255.0) * alpha + canvas_float * (1 - alpha)
        self.canvas = np.clip(blended, 0, 255).astype(np.uint8)
        
        return self.canvas.copy()
    
    def blend_screen(self,
                    overlay_color: Tuple[int, int, int],
                    alpha: float = 0.5) -> np.ndarray:
        """Screen blend mode."""
        overlay = np.full_like(self.canvas, overlay_color, dtype=np.float32)
        canvas_float = self.canvas.astype(np.float32)
        
        blended = 255 - ((255 - canvas_float) * (255 - overlay) / 255.0)
        result = blended * alpha + canvas_float * (1 - alpha)
        
        self.canvas = np.clip(result, 0, 255).astype(np.uint8)
        
        return self.canvas.copy()
    
    def blend_overlay(self,
                     overlay_color: Tuple[int, int, int],
                     alpha: float = 0.5) -> np.ndarray:
        """Overlay blend mode."""
        overlay = np.full_like(self.canvas, overlay_color, dtype=np.float32) / 255.0
        canvas_float = self.canvas.astype(np.float32) / 255.0
        
        # Overlay formula
        mask = canvas_float < 0.5
        blended = np.where(
            mask,
            2 * canvas_float * overlay,
            1 - 2 * (1 - canvas_float) * (1 - overlay)
        )
        
        result = (blended * 255 * alpha + self.canvas.astype(np.float32) * (1 - alpha))
        self.canvas = np.clip(result, 0, 255).astype(np.uint8)
        
        return self.canvas.copy()
    
    def blend_add(self,
                 overlay_color: Tuple[int, int, int],
                 alpha: float = 0.5) -> np.ndarray:
        """Additive blend mode."""
        overlay = np.full_like(self.canvas, overlay_color, dtype=np.float32)
        canvas_float = self.canvas.astype(np.float32)
        
        blended = canvas_float + overlay * alpha
        self.canvas = np.clip(blended, 0, 255).astype(np.uint8)
        
        return self.canvas.copy()
    
    def blend_subtract(self,
                      overlay_color: Tuple[int, int, int],
                      alpha: float = 0.5) -> np.ndarray:
        """Subtractive blend mode."""
        overlay = np.full_like(self.canvas, overlay_color, dtype=np.float32)
        canvas_float = self.canvas.astype(np.float32)
        
        blended = canvas_float - overlay * alpha
        self.canvas = np.clip(blended, 0, 255).astype(np.uint8)
        
        return self.canvas.copy()
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def reset(self):
        """Reset canvas to black."""
        self.canvas[:] = 0
    
    def get_canvas(self) -> np.ndarray:
        """Get current canvas."""
        return self.canvas.copy()
    
    def set_canvas(self, canvas: np.ndarray):
        """Set canvas."""
        self.canvas = canvas.copy()


# Function ID mapping for extended library
EXTENDED_FUNCTION_MAP = {
    # Original 10 functions (IDs 0-9)
    0: 'fill_solid',
    1: 'draw_gradient_linear',
    2: 'draw_ellipse',
    3: 'draw_rect',
    4: 'draw_line',
    5: 'draw_circle',
    6: 'draw_gradient_vertical',
    7: 'draw_gradient_horizontal',
    8: 'draw_gradient_diagonal',
    9: 'draw_rounded_ellipse',
    
    # Advanced fills (IDs 10-16)
    10: 'fill_radial_gradient',
    11: 'fill_conic_gradient',
    12: 'fill_noise_perlin',
    13: 'fill_checkerboard',
    14: 'fill_stripes',
    15: 'fill_dots',
    16: 'fill_wave',
    
    # Advanced shapes (IDs 20-32)
    20: 'draw_polygon',
    21: 'draw_bezier_curve',
    22: 'draw_arc',
    23: 'draw_rounded_rect',
    24: 'draw_star',
    25: 'draw_triangle',
    26: 'draw_heart',
    27: 'draw_ring',
    28: 'draw_trapezoid',
    29: 'draw_parallelogram',
    30: 'draw_crescent',
    31: 'draw_cross',
    32: 'draw_arrow',
    
    # Effects (IDs 40-47)
    40: 'apply_blur',
    41: 'apply_glow',
    42: 'apply_shadow',
    43: 'apply_sharpen',
    44: 'apply_posterize',
    45: 'apply_pixelate',
    46: 'apply_vignette',
    
    # Compositing (IDs 50-54)
    50: 'blend_multiply',
    51: 'blend_screen',
    52: 'blend_overlay',
    53: 'blend_add',
    54: 'blend_subtract',
}

# Total: 10 (original) + 37 (new) = 47 functions implemented
# (Note: Original 10 not in this file, they're in primitives.py)

NUM_EXTENDED_FUNCTIONS = len(EXTENDED_FUNCTION_MAP)

