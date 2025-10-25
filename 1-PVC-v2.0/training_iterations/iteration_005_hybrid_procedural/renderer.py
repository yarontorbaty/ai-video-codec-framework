"""
Helper function to render anime frames from operation sequences
"""
import numpy as np
import cv2

def render_anime_frame(operations, width, height):
    """
    Render an anime frame from a sequence of drawing operations
    
    Args:
        operations: List of (function_id, params) tuples
        width: Frame width
        height: Frame height
    
    Returns:
        frame: (H, W, 3) numpy array, uint8
    """
    # Import anime functions
    from anime_functions import (
        draw_anime_outline, draw_curved_outline, draw_tapered_line,
        fill_cel_region, add_cel_shadow, add_cel_highlight,
        draw_hair_gradient, draw_radial_gradient, draw_anime_eye,
        draw_anime_face_base, draw_anime_mouth, add_speed_lines,
        add_screen_tone, draw_sparkle_effect, draw_sky_gradient,
        draw_simple_cloud
    )
    
    # Function mapping
    function_map = {
        0: draw_anime_outline,
        1: draw_curved_outline,
        2: draw_tapered_line,
        3: fill_cel_region,
        4: add_cel_shadow,
        5: add_cel_highlight,
        6: draw_hair_gradient,
        7: draw_radial_gradient,
        8: draw_anime_eye,
        9: draw_anime_face_base,
        10: draw_anime_mouth,
        11: add_speed_lines,
        12: add_screen_tone,
        13: draw_sparkle_effect,
        14: draw_sky_gradient,
    }
    
    # Create blank canvas (white background) - FORCE exact dimensions
    frame = np.ones((int(height), int(width), 3), dtype=np.uint8) * 255
    
    # Execute operations
    for func_id, params in operations:
        if func_id >= len(function_map):
            continue
        
        func = function_map[func_id]
        
        # Convert normalized params [0,1] to actual values
        try:
            if func_id == 0:  # draw_anime_outline
                x1 = int(params[0] * width)
                y1 = int(params[1] * height)
                x2 = int(params[2] * width)
                y2 = int(params[3] * height)
                thickness = int(params[4] * 10) + 1
                color = (int(params[5]*255), int(params[6]*255), int(params[7]*255))
                func(frame, x1, y1, x2, y2, thickness, color)
            
            elif func_id in [3, 4, 5]:  # fill_cel_region, add_cel_shadow, add_cel_highlight
                # Generate polygon points
                num_points = 4
                points = []
                for i in range(num_points):
                    x = int(params[i*2] * width)
                    y = int(params[i*2+1] * height)
                    points.append([x, y])
                points = np.array(points, dtype=np.int32)
                color = (int(params[8]*255), int(params[9]*255), int(params[7]*255))
                
                if func_id == 3:
                    func(frame, points, color)
                elif func_id == 4:
                    func(frame, points, color, shadow_factor=0.7)
                else:
                    func(frame, points, color, highlight_factor=1.3)
            
            elif func_id == 6:  # draw_hair_gradient
                x = int(params[0] * width)
                y = int(params[1] * height)
                w = int(params[2] * width * 0.3)
                h = int(params[3] * height * 0.3)
                color1 = (int(params[4]*255), int(params[5]*255), int(params[6]*255))
                color2 = (int(params[7]*255), int(params[8]*255), int(params[9]*255))
                func(frame, x, y, w, h, color1, color2)
            
            elif func_id == 7:  # draw_radial_gradient
                cx = int(params[0] * width)
                cy = int(params[1] * height)
                inner_r = int(params[2] * min(width, height) * 0.2)
                outer_r = int(params[3] * min(width, height) * 0.4)
                inner_color = (int(params[4]*255), int(params[5]*255), int(params[6]*255))
                outer_color = (int(params[7]*255), int(params[8]*255), int(params[9]*255))
                func(frame, cx, cy, inner_r, outer_r, inner_color, outer_color)
            
            elif func_id == 8:  # draw_anime_eye
                cx = int(params[0] * width)
                cy = int(params[1] * height)
                w = int(params[2] * width * 0.1)
                h = int(params[3] * height * 0.1)
                eye_color = (int(params[4]*255), int(params[5]*255), int(params[6]*255))
                func(frame, cx, cy, w, h, eye_color)
            
            elif func_id == 9:  # draw_anime_face_base
                cx = int(params[0] * width)
                cy = int(params[1] * height)
                radius = int(params[2] * min(width, height) * 0.2)
                skin_color = (int(params[3]*255), int(params[4]*255), int(params[5]*255))
                func(frame, cx, cy, radius, skin_color)
            
            elif func_id == 11:  # add_speed_lines
                cx = int(params[0] * width)
                cy = int(params[1] * height)
                num_lines = int(params[2] * 30) + 10
                length = int(params[3] * min(width, height) * 0.5)
                func(frame, cx, cy, num_lines, length)
            
            elif func_id == 14:  # draw_sky_gradient
                horizon_y = int(params[0] * height)
                top_color = (int(params[1]*255), int(params[2]*255), int(params[3]*255))
                bottom_color = (int(params[4]*255), int(params[5]*255), int(params[6]*255))
                func(frame, horizon_y, top_color, bottom_color)
            
        except Exception as e:
            # Silently skip failed operations
            continue
    
    # CRITICAL: Ensure exact output size (in case any function modified it)
    if frame.shape != (int(height), int(width), 3):
        frame = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    
    return frame

