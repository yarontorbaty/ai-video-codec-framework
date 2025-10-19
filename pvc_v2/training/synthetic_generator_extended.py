#!/usr/bin/env python3
"""
Extended Synthetic Data Generator for PVC v2.0

Generates training data using all 47 graphics functions (10 original + 37 new).
Creates more diverse and complex scenes for better neural network training.
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from graphics.primitives import GraphicsPrimitives, FunctionCall
from graphics.primitives_extended import ExtendedGraphicsPrimitives, EXTENDED_FUNCTION_MAP


class ExtendedSyntheticGenerator:
    """
    Extended synthetic data generator using 47 graphics functions.
    
    Generates (frame, function_sequence) pairs with much greater diversity
    than the original 10-function generator.
    """
    
    def __init__(self, width: int = 256, height: int = 256):
        self.width = width
        self.height = height
        self.renderer_basic = GraphicsPrimitives(width, height)
        self.renderer_extended = ExtendedGraphicsPrimitives(width, height)
        
        # Function weights (higher = more likely to be selected)
        self.function_weights = {
            # Backgrounds (high weight)
            'fill_solid': 3,
            'fill_radial_gradient': 3,
            'fill_conic_gradient': 2,
            'draw_gradient_linear': 3,
            'fill_checkerboard': 1,
            'fill_stripes': 1,
            'fill_dots': 1,
            'fill_wave': 1,
            'fill_noise_perlin': 2,
            
            # Shapes (medium weight)
            'draw_rect': 2,
            'draw_ellipse': 2,
            'draw_circle': 2,
            'draw_rounded_rect': 2,
            'draw_polygon': 2,
            'draw_triangle': 2,
            'draw_star': 1,
            'draw_heart': 1,
            'draw_ring': 1,
            'draw_trapezoid': 1,
            'draw_parallelogram': 1,
            'draw_crescent': 1,
            'draw_cross': 1,
            'draw_arrow': 1,
            'draw_arc': 1,
            'draw_bezier_curve': 1,
            
            # Effects (low weight, applied to existing content)
            'apply_blur': 0.5,
            'apply_glow': 0.5,
            'apply_shadow': 0.5,
            'apply_sharpen': 0.3,
            'apply_posterize': 0.3,
            'apply_pixelate': 0.3,
            'apply_vignette': 0.3,
            
            # Compositing (low weight)
            'blend_multiply': 0.5,
            'blend_screen': 0.5,
            'blend_overlay': 0.5,
            'blend_add': 0.3,
            'blend_subtract': 0.3,
        }
    
    def generate_random_color(self) -> Tuple[int, int, int]:
        """Generate random RGB color."""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    def generate_scene(self, num_functions: int = 10) -> Tuple[np.ndarray, List[dict]]:
        """
        Generate a scene using random functions from the extended library.
        
        Args:
            num_functions: Number of functions to use (5-50)
            
        Returns:
            (rendered_frame, function_calls_list)
            Each function_call dict contains: {func_id, func_name, params, normalized_params}
        """
        self.renderer_extended.reset()
        function_calls = []
        
        # STEP 1: Always start with a background
        bg_functions = ['fill_solid', 'fill_radial_gradient', 'fill_conic_gradient',
                       'draw_gradient_linear', 'fill_checkerboard', 'fill_stripes', 'fill_wave']
        bg_func = random.choice(bg_functions)
        
        func_call = self._generate_background(bg_func)
        if func_call:
            function_calls.append(func_call)
        
        # STEP 2: Add shapes and effects
        remaining_functions = num_functions - 1  # -1 for background
        
        # Decide mix: 70% shapes, 20% effects, 10% compositing
        num_shapes = int(remaining_functions * 0.7)
        num_effects = int(remaining_functions * 0.2)
        num_composite = remaining_functions - num_shapes - num_effects
        
        # Add shapes
        shape_functions = [
            'draw_rect', 'draw_ellipse', 'draw_circle', 'draw_rounded_rect',
            'draw_polygon', 'draw_triangle', 'draw_star', 'draw_heart',
            'draw_ring', 'draw_trapezoid', 'draw_parallelogram',
            'draw_crescent', 'draw_cross', 'draw_arrow', 'draw_arc', 'draw_bezier_curve'
        ]
        
        for _ in range(num_shapes):
            shape_func = random.choice(shape_functions)
            func_call = self._generate_shape(shape_func)
            if func_call:
                function_calls.append(func_call)
        
        # Add effects
        effect_functions = ['apply_blur', 'apply_glow', 'apply_sharpen',
                           'apply_posterize', 'apply_pixelate', 'apply_vignette']
        
        for _ in range(num_effects):
            effect_func = random.choice(effect_functions)
            func_call = self._generate_effect(effect_func)
            if func_call:
                function_calls.append(func_call)
        
        # Add compositing
        composite_functions = ['blend_multiply', 'blend_screen', 'blend_overlay',
                              'blend_add', 'blend_subtract']
        
        for _ in range(num_composite):
            composite_func = random.choice(composite_functions)
            func_call = self._generate_composite(composite_func)
            if func_call:
                function_calls.append(func_call)
        
        # Get final rendered frame
        frame = self.renderer_extended.get_canvas()
        
        return frame, function_calls
    
    def _generate_background(self, func_name: str) -> dict:
        """Generate a background function call."""
        func_id = self._get_function_id(func_name)
        
        if func_name == 'fill_solid':
            color = self.generate_random_color()
            self.renderer_extended.canvas[:] = color
            return {
                'func_id': func_id,
                'func_name': func_name,
                'params': {'color': color},
                'normalized_params': self._normalize_params(color1=color)
            }
        
        elif func_name == 'fill_radial_gradient':
            cx, cy = self.width // 2, self.height // 2
            radius = max(self.width, self.height)
            color1 = self.generate_random_color()
            color2 = self.generate_random_color()
            self.renderer_extended.fill_radial_gradient(cx, cy, radius, color1, color2)
            return {
                'func_id': func_id,
                'func_name': func_name,
                'params': {'center_x': cx, 'center_y': cy, 'radius': radius,
                          'color_inner': color1, 'color_outer': color2},
                'normalized_params': self._normalize_params(
                    coords=(cx, cy, cx + radius, cy + radius),
                    color1=color1, color2=color2
                )
            }
        
        elif func_name == 'fill_conic_gradient':
            cx, cy = self.width // 2, self.height // 2
            color1 = self.generate_random_color()
            color2 = self.generate_random_color()
            self.renderer_extended.fill_conic_gradient(cx, cy, color1, color2)
            return {
                'func_id': func_id,
                'func_name': func_name,
                'params': {'center_x': cx, 'center_y': cy, 'color1': color1, 'color2': color2},
                'normalized_params': self._normalize_params(
                    coords=(cx, cy, cx, cy),
                    color1=color1, color2=color2
                )
            }
        
        elif func_name == 'fill_checkerboard':
            square_size = random.randint(8, 32)
            color1 = self.generate_random_color()
            color2 = self.generate_random_color()
            self.renderer_extended.fill_checkerboard(square_size, color1, color2)
            return {
                'func_id': func_id,
                'func_name': func_name,
                'params': {'square_size': square_size, 'color1': color1, 'color2': color2},
                'normalized_params': self._normalize_params(color1=color1, color2=color2)
            }
        
        # Add more background types as needed...
        return None
    
    def _generate_shape(self, func_name: str) -> dict:
        """Generate a shape function call."""
        func_id = self._get_function_id(func_name)
        color = self.generate_random_color()
        
        if func_name == 'draw_rect':
            x = random.randint(0, self.width - 50)
            y = random.randint(0, self.height - 50)
            w = random.randint(20, min(100, self.width - x))
            h = random.randint(20, min(100, self.height - y))
            # Simple filled rect (note: primitives.py function needs to be called differently)
            import cv2
            cv2.rectangle(self.renderer_extended.canvas, (x, y), (x + w, y + h), color, -1)
            return {
                'func_id': func_id,
                'func_name': func_name,
                'params': {'x': x, 'y': y, 'width': w, 'height': h, 'color': color},
                'normalized_params': self._normalize_params(coords=(x, y, x + w, y + h), color1=color)
            }
        
        elif func_name == 'draw_circle':
            cx = random.randint(30, self.width - 30)
            cy = random.randint(30, self.height - 30)
            radius = random.randint(10, 50)
            import cv2
            cv2.circle(self.renderer_extended.canvas, (cx, cy), radius, color, -1)
            return {
                'func_id': func_id,
                'func_name': func_name,
                'params': {'center_x': cx, 'center_y': cy, 'radius': radius, 'color': color},
                'normalized_params': self._normalize_params(
                    coords=(cx - radius, cy - radius, cx + radius, cy + radius),
                    color1=color
                )
            }
        
        elif func_name == 'draw_star':
            cx = random.randint(50, self.width - 50)
            cy = random.randint(50, self.height - 50)
            outer_r = random.randint(20, 50)
            inner_r = random.randint(10, outer_r - 5)
            points = random.choice([5, 6, 7, 8])
            self.renderer_extended.draw_star(cx, cy, outer_r, inner_r, points, color, True)
            return {
                'func_id': func_id,
                'func_name': func_name,
                'params': {'center_x': cx, 'center_y': cy, 'outer_radius': outer_r,
                          'inner_radius': inner_r, 'points': points, 'color': color},
                'normalized_params': self._normalize_params(
                    coords=(cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r),
                    color1=color
                )
            }
        
        # Add more shapes...
        return None
    
    def _generate_effect(self, func_name: str) -> dict:
        """Generate an effect function call."""
        func_id = self._get_function_id(func_name)
        
        if func_name == 'apply_blur':
            x = random.randint(0, self.width // 2)
            y = random.randint(0, self.height // 2)
            w = random.randint(50, self.width - x)
            h = random.randint(50, self.height - y)
            blur_size = random.choice([5, 9, 15, 21])
            self.renderer_extended.apply_blur(x, y, w, h, blur_size)
            return {
                'func_id': func_id,
                'func_name': func_name,
                'params': {'x': x, 'y': y, 'width': w, 'height': h, 'blur_size': blur_size},
                'normalized_params': self._normalize_params(coords=(x, y, x + w, y + h))
            }
        
        elif func_name == 'apply_vignette':
            intensity = random.uniform(0.3, 0.7)
            self.renderer_extended.apply_vignette(intensity)
            return {
                'func_id': func_id,
                'func_name': func_name,
                'params': {'intensity': intensity},
                'normalized_params': self._normalize_params()
            }
        
        return None
    
    def _generate_composite(self, func_name: str) -> dict:
        """Generate a compositing function call."""
        func_id = self._get_function_id(func_name)
        color = self.generate_random_color()
        alpha = random.uniform(0.3, 0.7)
        
        if func_name == 'blend_multiply':
            self.renderer_extended.blend_multiply(color, alpha)
        elif func_name == 'blend_screen':
            self.renderer_extended.blend_screen(color, alpha)
        elif func_name == 'blend_overlay':
            self.renderer_extended.blend_overlay(color, alpha)
        elif func_name == 'blend_add':
            self.renderer_extended.blend_add(color, alpha)
        elif func_name == 'blend_subtract':
            self.renderer_extended.blend_subtract(color, alpha)
        
        return {
            'func_id': func_id,
            'func_name': func_name,
            'params': {'overlay_color': color, 'alpha': alpha},
            'normalized_params': self._normalize_params(color1=color)
        }
    
    def _get_function_id(self, func_name: str) -> int:
        """Get function ID from name."""
        for fid, fname in EXTENDED_FUNCTION_MAP.items():
            if fname == func_name:
                return fid
        return 0  # Default to fill_solid
    
    def _normalize_params(self, 
                         coords: Tuple[int, int, int, int] = None,
                         color1: Tuple[int, int, int] = None,
                         color2: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Normalize parameters to [0,1] range for neural network.
        Returns array of shape (10,): [coords(4), color1(3), color2(3)]
        """
        result = []
        
        # Coordinates
        if coords:
            result.extend([
                coords[0] / self.width,
                coords[1] / self.height,
                coords[2] / self.width,
                coords[3] / self.height
            ])
        else:
            result.extend([0.5, 0.5, 0.5, 0.5])
        
        # Color 1
        if color1:
            result.extend([c / 255.0 for c in color1])
        else:
            result.extend([0.5, 0.5, 0.5])
        
        # Color 2
        if color2:
            result.extend([c / 255.0 for c in color2])
        else:
            result.extend([0.5, 0.5, 0.5])
        
        return np.array(result, dtype=np.float32)
    
    def generate_dataset(self, 
                        num_samples: int = 1000,
                        min_functions: int = 5,
                        max_functions: int = 20) -> Tuple[np.ndarray, List[List[dict]]]:
        """
        Generate a complete dataset.
        
        Args:
            num_samples: Number of (frame, sequence) pairs to generate
            min_functions: Minimum functions per scene
            max_functions: Maximum functions per scene
            
        Returns:
            (frames_array, function_sequences_list)
            frames_array: shape (num_samples, height, width, 3)
            function_sequences_list: list of function call lists
        """
        print(f"Generating {num_samples} samples with {min_functions}-{max_functions} functions each...")
        
        frames = []
        sequences = []
        
        for i in range(num_samples):
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples...")
            
            num_funcs = random.randint(min_functions, max_functions)
            frame, func_seq = self.generate_scene(num_funcs)
            
            frames.append(frame)
            sequences.append(func_seq)
        
        frames_array = np.array(frames, dtype=np.uint8)
        print(f"✅ Generated {num_samples} samples")
        print(f"   Frame shape: {frames_array.shape}")
        print(f"   Avg functions/sample: {np.mean([len(s) for s in sequences]):.1f}")
        
        return frames_array, sequences


if __name__ == "__main__":
    # Quick test
    gen = ExtendedSyntheticGenerator(width=256, height=256)
    frame, funcs = gen.generate_scene(num_functions=10)
    
    print(f"Generated scene with {len(funcs)} functions")
    print(f"Frame shape: {frame.shape}")
    print(f"Functions used:")
    for i, fc in enumerate(funcs):
        print(f"  {i+1}. {fc['func_name']} (ID {fc['func_id']})")

