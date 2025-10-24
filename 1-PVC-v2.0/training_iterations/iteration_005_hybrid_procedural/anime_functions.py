"""
Anime-Specific Procedural Graphics Functions
For Hybrid Neural-Procedural Image Compression

These functions are specifically designed for anime/animation style:
- Thick outlines (line art)
- Cel shading (flat color regions)
- Gradient fills (hair, clothing)
- Character-specific shapes (eyes, faces, bodies)
- Background elements (clouds, sky gradients, patterns)
"""

import numpy as np
import cv2
from typing import Tuple, List

# ============================================================================
# ANIME LINE ART FUNCTIONS
# ============================================================================

def draw_anime_outline(img, x1, y1, x2, y2, thickness=3, color=(0, 0, 0)):
    """
    Draw thick anime-style outline
    Default: Black, 3px thick (typical anime line weight)
    """
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
    return img


def draw_curved_outline(img, points, thickness=3, color=(0, 0, 0), closed=False):
    """
    Draw smooth curved outline using Bezier/spline
    Used for: Character silhouettes, hair curves, clothing folds
    """
    points = np.array(points, dtype=np.int32)
    cv2.polylines(img, [points], closed, color, thickness, cv2.LINE_AA)
    return img


def draw_tapered_line(img, x1, y1, x2, y2, start_thick=5, end_thick=1, color=(0, 0, 0)):
    """
    Line that tapers from thick to thin
    Used for: Hair strands, motion lines, emphasis lines
    """
    # Implementation: draw multiple segments with decreasing thickness
    steps = 20
    for i in range(steps):
        t = i / steps
        x = int(x1 + (x2 - x1) * t)
        y = int(y1 + (y2 - y1) * t)
        x_next = int(x1 + (x2 - x1) * (t + 1/steps))
        y_next = int(y1 + (y2 - y1) * (t + 1/steps))
        thick = int(start_thick + (end_thick - start_thick) * t)
        cv2.line(img, (x, y), (x_next, y_next), color, max(1, thick), cv2.LINE_AA)
    return img


# ============================================================================
# CEL SHADING & FLAT COLOR REGIONS
# ============================================================================

def fill_cel_region(img, points, color):
    """
    Fill a region with flat color (cel shading)
    Used for: Skin, clothing, hair base color
    """
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(img, [points], color, cv2.LINE_AA)
    return img


def add_cel_shadow(img, points, base_color, shadow_factor=0.7):
    """
    Add cel-shaded shadow (darker flat region)
    Used for: Character shadows, depth
    """
    shadow_color = tuple(int(c * shadow_factor) for c in base_color)
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(img, [points], shadow_color, cv2.LINE_AA)
    return img


def add_cel_highlight(img, points, base_color, highlight_factor=1.3):
    """
    Add cel-shaded highlight (lighter flat region)
    Used for: Shiny hair, reflections, glossy surfaces
    """
    highlight_color = tuple(min(255, int(c * highlight_factor)) for c in base_color)
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(img, [points], highlight_color, cv2.LINE_AA)
    return img


# ============================================================================
# ANIME-SPECIFIC GRADIENTS
# ============================================================================

def draw_hair_gradient(img, x, y, width, height, color1, color2, angle=90):
    """
    Smooth gradient for anime hair
    Typically: lighter at top, darker at bottom
    """
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        t = i / height
        color = tuple(int(c1 * (1-t) + c2 * t) for c1, c2 in zip(color1, color2))
        gradient[i, :] = color
    
    # Apply to region
    y1, y2 = max(0, y), min(img.shape[0], y + height)
    x1, x2 = max(0, x), min(img.shape[1], x + width)
    img[y1:y2, x1:x2] = gradient[:y2-y1, :x2-x1]
    return img


def draw_radial_gradient(img, cx, cy, inner_radius, outer_radius, inner_color, outer_color):
    """
    Radial gradient
    Used for: Eyes (iris), cheek blush, light effects
    """
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Normalize distance
    mask = (dist >= inner_radius) & (dist <= outer_radius)
    t = np.clip((dist - inner_radius) / (outer_radius - inner_radius), 0, 1)
    
    for c in range(3):
        img[:, :, c] = np.where(mask, 
                                 inner_color[c] * (1-t) + outer_color[c] * t,
                                 img[:, :, c])
    return img


# ============================================================================
# ANIME CHARACTER PRIMITIVES
# ============================================================================

def draw_anime_eye(img, cx, cy, width, height, eye_color, highlight_color=(255, 255, 255)):
    """
    Draw simplified anime eye
    - Oval shape
    - Colored iris with gradient
    - White highlight (catchlight)
    - Black outline
    """
    # Outer oval (white of eye)
    cv2.ellipse(img, (cx, cy), (width, height), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
    
    # Iris (colored gradient)
    iris_w, iris_h = int(width * 0.7), int(height * 0.7)
    draw_radial_gradient(img, cx, cy, 0, iris_h, eye_color, tuple(int(c*0.7) for c in eye_color))
    
    # Highlight (anime signature)
    highlight_x, highlight_y = cx - width//3, cy - height//3
    cv2.circle(img, (highlight_x, highlight_y), width//4, highlight_color, -1, cv2.LINE_AA)
    
    # Outline
    cv2.ellipse(img, (cx, cy), (width, height), 0, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
    
    return img


def draw_anime_face_base(img, cx, cy, radius, skin_color):
    """
    Draw base anime face circle
    """
    cv2.circle(img, (cx, cy), radius, skin_color, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), radius, (0, 0, 0), 2, cv2.LINE_AA)  # Outline
    return img


def draw_anime_mouth(img, cx, cy, width, smile_curve=0.5, color=(0, 0, 0)):
    """
    Draw anime mouth
    smile_curve: 0 = neutral, 1 = big smile, -1 = frown
    """
    # Simple arc
    start_angle = 0 if smile_curve > 0 else 180
    end_angle = 180 if smile_curve > 0 else 360
    axes = (width, int(width * abs(smile_curve) * 0.3))
    cv2.ellipse(img, (cx, cy), axes, 0, start_angle, end_angle, color, 2, cv2.LINE_AA)
    return img


# ============================================================================
# ANIME EFFECTS & DETAILS
# ============================================================================

def add_speed_lines(img, center_x, center_y, num_lines=20, length=200, thickness=2):
    """
    Add motion/speed lines radiating from center
    Classic anime action effect
    """
    for i in range(num_lines):
        angle = (i / num_lines) * 2 * np.pi
        x1 = int(center_x + length * 0.3 * np.cos(angle))
        y1 = int(center_y + length * 0.3 * np.sin(angle))
        x2 = int(center_x + length * np.cos(angle))
        y2 = int(center_y + length * np.sin(angle))
        draw_tapered_line(img, x1, y1, x2, y2, thickness, 1, (0, 0, 0))
    return img


def add_screen_tone(img, x, y, width, height, pattern='dots', density=0.5):
    """
    Add manga/anime screen tone pattern
    Used for: Backgrounds, shading, texture
    """
    if pattern == 'dots':
        spacing = int(10 / density)
        for i in range(y, y + height, spacing):
            for j in range(x, x + width, spacing):
                if i < img.shape[0] and j < img.shape[1]:
                    radius = int(3 * density)
                    cv2.circle(img, (j, i), radius, (0, 0, 0), -1, cv2.LINE_AA)
    elif pattern == 'lines':
        spacing = int(8 / density)
        for i in range(y, y + height, spacing):
            if i < img.shape[0]:
                cv2.line(img, (x, i), (x + width, i), (0, 0, 0), 1, cv2.LINE_AA)
    return img


def draw_sparkle_effect(img, cx, cy, size=20, color=(255, 255, 200)):
    """
    Draw anime sparkle/star effect
    Used for: Eyes, magical effects, emphasis
    """
    # Four-pointed star
    points = [
        (cx, cy - size),  # Top
        (cx + size//3, cy - size//3),
        (cx + size, cy),  # Right
        (cx + size//3, cy + size//3),
        (cx, cy + size),  # Bottom
        (cx - size//3, cy + size//3),
        (cx - size, cy),  # Left
        (cx - size//3, cy - size//3),
    ]
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(img, [points], color, cv2.LINE_AA)
    return img


# ============================================================================
# ANIME BACKGROUND ELEMENTS
# ============================================================================

def draw_sky_gradient(img, horizon_y, sky_color_top, sky_color_bottom):
    """
    Draw typical anime sky gradient
    """
    for y in range(horizon_y):
        t = y / horizon_y
        color = tuple(int(c1 * (1-t) + c2 * t) for c1, c2 in zip(sky_color_top, sky_color_bottom))
        img[y, :] = color
    return img


def draw_simple_cloud(img, cx, cy, width, height, color=(255, 255, 255)):
    """
    Draw simplified anime-style cloud (overlapping circles)
    """
    num_puffs = 3
    for i in range(num_puffs):
        x_offset = int((i - num_puffs/2) * width / num_puffs)
        radius = int(height / 2)
        cv2.circle(img, (cx + x_offset, cy), radius, color, -1, cv2.LINE_AA)
    return img


# ============================================================================
# FUNCTION REGISTRY
# ============================================================================

ANIME_FUNCTIONS = {
    # Line art (0-9)
    0: ('anime_outline', draw_anime_outline),
    1: ('curved_outline', draw_curved_outline),
    2: ('tapered_line', draw_tapered_line),
    
    # Cel shading (10-19)
    10: ('cel_region', fill_cel_region),
    11: ('cel_shadow', add_cel_shadow),
    12: ('cel_highlight', add_cel_highlight),
    
    # Gradients (20-29)
    20: ('hair_gradient', draw_hair_gradient),
    21: ('radial_gradient', draw_radial_gradient),
    
    # Character features (30-39)
    30: ('anime_eye', draw_anime_eye),
    31: ('anime_face', draw_anime_face_base),
    32: ('anime_mouth', draw_anime_mouth),
    
    # Effects (40-49)
    40: ('speed_lines', add_speed_lines),
    41: ('screen_tone', add_screen_tone),
    42: ('sparkle', draw_sparkle_effect),
    
    # Backgrounds (50-59)
    50: ('sky_gradient', draw_sky_gradient),
    51: ('simple_cloud', draw_simple_cloud),
}

NUM_ANIME_FUNCTIONS = len(ANIME_FUNCTIONS)

