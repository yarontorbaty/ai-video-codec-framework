"""
Scene Renderer - PVC Decoder Module

Reconstructs video frames from procedural scene descriptions (ISP).
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from .procedural_textures import ProceduralTextures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SceneRenderer:
    """
    Renders frames from procedural scene descriptions.
    
    Takes an Intermediate Scene Program (ISP) and generates video frames
    by drawing contours, applying procedural textures, and animating motion.
    """
    
    def __init__(self, resolution: Tuple[int, int] = (1280, 720)):
        """
        Initialize renderer.
        
        Args:
            resolution: (width, height) of output frames
        """
        self.width, self.height = resolution
        self.texture_gen = ProceduralTextures()
    
    def render_frame(self,
                    frame_idx: int,
                    scene_desc: Dict) -> np.ndarray:
        """
        Render a single frame from scene description.
        
        Args:
            frame_idx: Frame index to render
            scene_desc: Scene description with objects, textures, motion
            
        Returns:
            Rendered frame (H x W x 3) in BGR format
        """
        # Create blank canvas
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Get objects for this frame
        objects = scene_desc.get('objects', [])
        
        # Render each object
        for obj in objects:
            self._render_object(frame, obj, frame_idx)
        
        # Apply residuals if present
        residuals = scene_desc.get('residuals', [])
        for residual in residuals:
            if residual.get('frame_idx') == frame_idx:
                self._apply_residual(frame, residual)
        
        return frame
    
    def _render_object(self,
                      frame: np.ndarray,
                      obj: Dict,
                      frame_idx: int):
        """
        Render a single object onto the frame.
        
        Args:
            frame: Frame to render into (modified in-place)
            obj: Object dictionary with contours, texture, motion
            frame_idx: Current frame index
        """
        # Get contour for this frame
        contour_data = self._get_contour_at_frame(obj, frame_idx)
        if contour_data is None:
            return
        
        # Get motion for this frame
        position = self._get_position_at_frame(obj, frame_idx)
        
        # Convert contour points to numpy array
        points = np.array(contour_data['points'], dtype=np.int32)
        
        # Apply motion (translation)
        points[:, 0] += int(position[0])
        points[:, 1] += int(position[1])
        
        # Clip to frame bounds
        points[:, 0] = np.clip(points[:, 0], 0, self.width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, self.height - 1)
        
        # Get texture
        texture_info = obj.get('texture', {})
        color = self._get_object_color(texture_info, contour_data)
        
        # Draw filled contour
        cv2.fillPoly(frame, [points], color=color)
        
        # Draw contour outline for sharp edges
        cv2.polylines(frame, [points], 
                     isClosed=True, 
                     color=color,
                     thickness=1)
    
    def _get_contour_at_frame(self,
                             obj: Dict,
                             frame_idx: int) -> Optional[Dict]:
        """
        Get contour data for a specific frame.
        
        Uses keyframes and interpolation if needed.
        """
        contour_seq = obj.get('contour_sequence', [])
        keyframes = obj.get('keyframes', [0])
        
        if not contour_seq:
            return None
        
        # If frame_idx is before object appears
        if frame_idx < keyframes[0]:
            return None
        
        # If frame_idx is after object disappears
        if frame_idx >= len(contour_seq):
            return contour_seq[-1]
        
        # Return contour at this frame
        return contour_seq[min(frame_idx, len(contour_seq) - 1)]
    
    def _get_position_at_frame(self,
                              obj: Dict,
                              frame_idx: int) -> Tuple[float, float]:
        """
        Get object position at a specific frame.
        
        Interpolates motion from motion model.
        """
        motion_seq = obj.get('motion_sequence', [])
        
        if not motion_seq:
            return (0.0, 0.0)
        
        # Clamp frame index
        idx = min(frame_idx, len(motion_seq) - 1)
        
        if idx < 0:
            return (0.0, 0.0)
        
        # Accumulate translations
        total_dx = sum(m['translation'][0] for m in motion_seq[:idx + 1])
        total_dy = sum(m['translation'][1] for m in motion_seq[:idx + 1])
        
        return (total_dx, total_dy)
    
    def _get_object_color(self,
                         texture_info: Dict,
                         contour_data: Dict) -> Tuple[int, int, int]:
        """
        Get color for an object based on its texture.
        
        For simple prototype, we'll use solid colors or simple patterns.
        Full implementation would generate texture patches.
        """
        texture_type = texture_info.get('type', 'solid')
        
        if texture_type == 'solid':
            # Solid color (stored as normalized RGB)
            rgb = texture_info.get('color', [0.5, 0.5, 0.5])
            return (
                int(rgb[2] * 255),  # B
                int(rgb[1] * 255),  # G
                int(rgb[0] * 255)   # R
            )
        
        elif texture_type in ['perlin', 'worley', 'fbm']:
            # For procedural textures, we'd ideally generate a texture patch
            # and fill the contour with it. For simplicity, use avg color.
            # This is a simplified version - full impl would be more sophisticated
            
            # Generate small sample
            sample = self.texture_gen.generate_texture(
                texture_type,
                shape=(16, 16),
                params=texture_info.get('params', {})
            )
            avg_value = np.mean(sample)
            
            # Map to color
            color_map = texture_info.get('color_map', [avg_value, avg_value, avg_value])
            return (
                int(color_map[2] * 255),
                int(color_map[1] * 255),
                int(color_map[0] * 255)
            )
        
        else:
            # Default gray
            return (128, 128, 128)
    
    def _apply_residual(self,
                       frame: np.ndarray,
                       residual: Dict):
        """
        Apply residual correction to frame.
        
        Residuals are stored as compressed tiles for high-entropy regions.
        """
        # Get tile position
        x, y = residual.get('position', (0, 0))
        w, h = residual.get('size', (0, 0))
        
        # Get residual data (would be AV1-encoded in practice)
        tile_data = residual.get('data', None)
        
        if tile_data is None:
            return
        
        # For prototype, assume tile_data is a numpy array
        # In production, would decode from compressed format
        if isinstance(tile_data, np.ndarray):
            # Clip to frame bounds
            x2 = min(x + w, self.width)
            y2 = min(y + h, self.height)
            
            if x2 > x and y2 > y:
                # Resize tile if needed
                tile_h, tile_w = tile_data.shape[:2]
                if tile_h != (y2 - y) or tile_w != (x2 - x):
                    tile_data = cv2.resize(tile_data, (x2 - x, y2 - y))
                
                # Add residual
                frame[y:y2, x:x2] = tile_data
    
    def render_sequence(self,
                       scene_desc: Dict,
                       num_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Render a sequence of frames.
        
        Args:
            scene_desc: Complete scene description
            num_frames: Number of frames to render (None = auto-detect)
            
        Returns:
            List of rendered frames
        """
        if num_frames is None:
            # Detect from metadata or objects
            metadata = scene_desc.get('metadata', {})
            duration = metadata.get('duration', 1.0)
            fps = metadata.get('fps', 30)
            num_frames = int(duration * fps)
        
        logger.info(f"Rendering {num_frames} frames...")
        
        frames = []
        for i in range(num_frames):
            frame = self.render_frame(i, scene_desc)
            frames.append(frame)
            
            if (i + 1) % 30 == 0:
                logger.info(f"  Rendered {i + 1}/{num_frames} frames")
        
        logger.info(f"✅ Rendered {num_frames} frames")
        return frames
    
    def save_video(self,
                  frames: List[np.ndarray],
                  output_path: str,
                  fps: int = 30,
                  codec: str = 'mp4v'):
        """
        Save rendered frames as video file.
        
        Args:
            frames: List of BGR frames
            output_path: Output video path
            fps: Frames per second
            codec: Video codec (e.g., 'mp4v', 'avc1')
        """
        if not frames:
            logger.error("No frames to save")
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        logger.info(f"✅ Saved video: {output_path}")


# Example usage
if __name__ == "__main__":
    # Create a simple test scene
    scene = {
        'metadata': {
            'resolution': [1280, 720],
            'fps': 30,
            'duration': 1.0
        },
        'objects': [
            {
                'id': 0,
                'contour_sequence': [
                    {
                        'points': [[100, 100], [300, 100], [300, 300], [100, 300]],
                        'centroid': [200, 200]
                    }
                ] * 30,  # Static for 30 frames
                'motion_sequence': [
                    {'translation': [2.0, 1.0], 'rotation': 0.0, 'scale': 1.0}
                ] * 29,
                'keyframes': [0],
                'texture': {
                    'type': 'solid',
                    'color': [0.2, 0.6, 0.9]  # RGB
                }
            }
        ],
        'residuals': []
    }
    
    # Render
    renderer = SceneRenderer(resolution=(640, 480))
    frames = renderer.render_sequence(scene, num_frames=30)
    
    print(f"✅ Rendered {len(frames)} frames")
    print(f"   Frame shape: {frames[0].shape}")

