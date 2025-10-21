#!/usr/bin/env python3
"""
Synthetic Data Generator for PVC v2.0

Generates training pairs: (rendered_frame, function_sequence)
This provides ground truth for training the neural network.
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple
import random

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphics.primitives import GraphicsPrimitives, FunctionCall


class SyntheticDataGenerator:
    """
    Generates synthetic training data for PVC v2.0.
    
    Creates random function sequences and renders them to create
    (frame, function_sequence) pairs for neural network training.
    """
    
    def __init__(self, width: int = 256, height: int = 256):
        """
        Initialize generator.
        
        Args:
            width, height: Output frame size (use smaller for faster training)
        """
        self.width = width
        self.height = height
        self.renderer = GraphicsPrimitives(width, height)
    
    def generate_random_color(self) -> Tuple[float, float, float]:
        """Generate a random color."""
        return (random.random(), random.random(), random.random())
    
    def generate_simple_scene(self, complexity: int = 5) -> Tuple[np.ndarray, List[FunctionCall]]:
        """
        Generate a simple synthetic scene.
        
        Args:
            complexity: Number of function calls (5-20 for training)
            
        Returns:
            (rendered_frame, function_calls)
        """
        self.renderer.create_canvas()
        function_calls = []
        
        # Always start with a background
        bg_type = random.choice(['solid', 'gradient_linear', 'gradient_radial'])
        
        if bg_type == 'solid':
            call = self.renderer.fill_solid(
                0, 0, self.width, self.height,
                self.generate_random_color()
            )
            function_calls.append(call)
        
        elif bg_type == 'gradient_linear':
            call = self.renderer.draw_gradient_linear(
                0, 0, self.width, self.height,
                self.generate_random_color(),
                self.generate_random_color(),
                angle=random.uniform(0, 180)
            )
            function_calls.append(call)
        
        else:  # gradient_radial
            call = self.renderer.draw_gradient_radial(
                self.width // 2, self.height // 2,
                max(self.width, self.height),
                self.generate_random_color(),
                self.generate_random_color()
            )
            function_calls.append(call)
        
        # Add random shapes
        num_shapes = complexity - 1  # -1 for background
        
        for _ in range(num_shapes):
            shape_type = random.choice(['rectangle', 'ellipse', 'polygon'])
            
            if shape_type == 'rectangle':
                x = random.randint(0, self.width - 20)
                y = random.randint(0, self.height - 20)
                w = random.randint(10, min(100, self.width - x))
                h = random.randint(10, min(100, self.height - y))
                
                use_fill = random.random() > 0.3
                use_stroke = random.random() > 0.7
                
                call = self.renderer.draw_rectangle(
                    x, y, w, h,
                    fill=self.generate_random_color() if use_fill else None,
                    stroke=self.generate_random_color() if use_stroke else None,
                    stroke_width=random.randint(1, 3) if use_stroke else 1,
                    opacity=random.uniform(0.5, 1.0)
                )
                function_calls.append(call)
            
            elif shape_type == 'ellipse':
                cx = random.randint(20, self.width - 20)
                cy = random.randint(20, self.height - 20)
                rx = random.randint(10, 50)
                ry = random.randint(10, 50)
                
                use_fill = random.random() > 0.3
                use_stroke = random.random() > 0.7
                
                call = self.renderer.draw_ellipse(
                    cx, cy, rx, ry,
                    fill=self.generate_random_color() if use_fill else None,
                    stroke=self.generate_random_color() if use_stroke else None,
                    stroke_width=random.randint(1, 3) if use_stroke else 1,
                    opacity=random.uniform(0.5, 1.0)
                )
                function_calls.append(call)
            
            else:  # polygon
                num_points = random.randint(3, 6)
                center_x = random.randint(50, self.width - 50)
                center_y = random.randint(50, self.height - 50)
                radius = random.randint(20, 50)
                
                points = []
                for i in range(num_points):
                    angle = (2 * np.pi * i) / num_points + random.uniform(-0.3, 0.3)
                    x = center_x + int(radius * np.cos(angle))
                    y = center_y + int(radius * np.sin(angle))
                    points.append((x, y))
                
                use_fill = random.random() > 0.3
                use_stroke = random.random() > 0.7
                
                call = self.renderer.draw_polygon(
                    points,
                    fill=self.generate_random_color() if use_fill else None,
                    stroke=self.generate_random_color() if use_stroke else None,
                    stroke_width=random.randint(1, 3) if use_stroke else 1,
                    opacity=random.uniform(0.5, 1.0)
                )
                function_calls.append(call)
        
        # Optionally add effects
        if random.random() > 0.7:
            # Add blur to random region
            x = random.randint(0, self.width // 2)
            y = random.randint(0, self.height // 2)
            w = random.randint(self.width // 4, self.width - x)
            h = random.randint(self.height // 4, self.height - y)
            
            call = self.renderer.apply_gaussian_blur(
                x, y, w, h,
                radius=random.randint(1, 5)
            )
            function_calls.append(call)
        
        if random.random() > 0.8:
            # Add noise to random region
            x = random.randint(0, self.width // 2)
            y = random.randint(0, self.height // 2)
            w = random.randint(self.width // 4, self.width - x)
            h = random.randint(self.height // 4, self.height - y)
            
            call = self.renderer.apply_noise(
                x, y, w, h,
                strength=random.uniform(0.05, 0.15),
                seed=random.randint(0, 10000)
            )
            function_calls.append(call)
        
        # Get rendered frame
        frame = self.renderer.get_canvas()
        
        return frame, function_calls
    
    def generate_anime_like_scene(self) -> Tuple[np.ndarray, List[FunctionCall]]:
        """
        Generate a scene that looks more like anime (simplified character).
        
        Returns:
            (rendered_frame, function_calls)
        """
        self.renderer.create_canvas()
        function_calls = []
        
        # Sky gradient background
        call = self.renderer.draw_gradient_linear(
            0, 0, self.width, self.height,
            (0.6, 0.8, 0.95),  # Light blue
            (0.4, 0.6, 0.85),  # Darker blue
            angle=90
        )
        function_calls.append(call)
        
        # Simple character: circle face + eyes + mouth
        face_cx = self.width // 2
        face_cy = self.height // 2
        face_r = self.width // 4
        
        # Face
        call = self.renderer.draw_ellipse(
            face_cx, face_cy, face_r, int(face_r * 1.2),
            fill=(0.95, 0.85, 0.75),  # Skin tone
            stroke=(0.3, 0.2, 0.1),   # Dark outline
            stroke_width=2,
            opacity=1.0
        )
        function_calls.append(call)
        
        # Left eye
        call = self.renderer.draw_ellipse(
            face_cx - face_r // 3, face_cy - face_r // 4,
            face_r // 6, face_r // 5,
            fill=(1.0, 1.0, 1.0),  # White
            stroke=(0.0, 0.0, 0.0),  # Black
            stroke_width=1
        )
        function_calls.append(call)
        
        # Left pupil
        call = self.renderer.draw_ellipse(
            face_cx - face_r // 3, face_cy - face_r // 4,
            face_r // 12, face_r // 10,
            fill=(0.2, 0.1, 0.0),  # Dark brown
            opacity=1.0
        )
        function_calls.append(call)
        
        # Right eye
        call = self.renderer.draw_ellipse(
            face_cx + face_r // 3, face_cy - face_r // 4,
            face_r // 6, face_r // 5,
            fill=(1.0, 1.0, 1.0),
            stroke=(0.0, 0.0, 0.0),
            stroke_width=1
        )
        function_calls.append(call)
        
        # Right pupil
        call = self.renderer.draw_ellipse(
            face_cx + face_r // 3, face_cy - face_r // 4,
            face_r // 12, face_r // 10,
            fill=(0.2, 0.1, 0.0),
            opacity=1.0
        )
        function_calls.append(call)
        
        # Mouth (small ellipse)
        call = self.renderer.draw_ellipse(
            face_cx, face_cy + face_r // 3,
            face_r // 8, face_r // 12,
            fill=(0.8, 0.3, 0.3),  # Red
            opacity=1.0
        )
        function_calls.append(call)
        
        # Hair (simple polygon on top)
        hair_points = [
            (face_cx - face_r, face_cy - face_r),
            (face_cx, face_cy - int(face_r * 1.5)),
            (face_cx + face_r, face_cy - face_r),
            (face_cx + int(face_r * 0.8), face_cy - int(face_r * 0.3)),
            (face_cx - int(face_r * 0.8), face_cy - int(face_r * 0.3)),
        ]
        call = self.renderer.draw_polygon(
            hair_points,
            fill=(0.2, 0.1, 0.05),  # Dark brown/black
            stroke=(0.1, 0.05, 0.0),
            stroke_width=2,
            opacity=1.0
        )
        function_calls.append(call)
        
        # Get rendered frame
        frame = self.renderer.get_canvas()
        
        return frame, function_calls
    
    def generate_dataset(self, num_samples: int = 1000, anime_ratio: float = 0.3) -> Tuple[List[np.ndarray], List[List[FunctionCall]]]:
        """
        Generate a full training dataset.
        
        Args:
            num_samples: Number of samples to generate
            anime_ratio: Fraction of samples that are anime-like (vs random)
            
        Returns:
            (frames, function_sequences)
        """
        frames = []
        function_sequences = []
        
        num_anime = int(num_samples * anime_ratio)
        num_random = num_samples - num_anime
        
        print(f"Generating {num_samples} training samples...")
        print(f"  {num_anime} anime-like scenes")
        print(f"  {num_random} random scenes")
        print()
        
        # Generate anime-like scenes
        for i in range(num_anime):
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_anime} anime scenes...")
            
            frame, funcs = self.generate_anime_like_scene()
            frames.append(frame)
            function_sequences.append(funcs)
        
        # Generate random scenes
        for i in range(num_random):
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_random} random scenes...")
            
            complexity = random.randint(5, 15)
            frame, funcs = self.generate_simple_scene(complexity)
            frames.append(frame)
            function_sequences.append(funcs)
        
        print(f"\n✅ Generated {num_samples} training samples!")
        
        return frames, function_sequences


if __name__ == '__main__':
    """Test the synthetic data generator."""
    import cv2
    
    generator = SyntheticDataGenerator(width=256, height=256)
    
    # Generate a few test samples
    print("Generating test samples...")
    
    for i in range(5):
        if i < 3:
            frame, funcs = generator.generate_anime_like_scene()
            name = f"anime_{i}"
        else:
            frame, funcs = generator.generate_simple_scene(complexity=10)
            name = f"random_{i-3}"
        
        # Save frame
        cv2.imwrite(f"/tmp/pvc_v2_test_{name}.png", frame)
        
        # Print function sequence
        total_bytes = sum(f.estimate_size() for f in funcs)
        print(f"\n{name}:")
        print(f"  Functions: {len(funcs)}")
        print(f"  Est. size: {total_bytes} bytes")
        print(f"  Compression: {(1 - total_bytes / (256*256*3)) * 100:.1f}%")
        
        for func in funcs[:3]:  # Show first 3
            print(f"    {func.func_name}({list(func.params.keys())})")
    
    print("\n✅ Test samples saved to /tmp/pvc_v2_test_*.png")

