"""
Motion Tracker - PVC Encoder Module

Tracks object motion across frames using optical flow and fits smooth motion functions.
"""

import cv2
import numpy as np
from scipy import interpolate
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MotionTracker:
    """
    Tracks motion of contours/objects across video frames.
    
    Uses optical flow to compute motion vectors, then fits smooth functions
    (splines, affine transforms) to represent motion compactly.
    """
    
    def __init__(self, 
                 flow_method: str = 'farneback',
                 keyframe_interval: int = 30):
        """
        Initialize motion tracker.
        
        Args:
            flow_method: 'farneback' or 'lucas_kanade'
            keyframe_interval: Store motion parameters every N frames
        """
        self.flow_method = flow_method
        self.keyframe_interval = keyframe_interval
    
    def compute_optical_flow(self, 
                            frame1: np.ndarray, 
                            frame2: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow between two frames.
        
        Args:
            frame1: First frame (grayscale or BGR)
            frame2: Second frame (grayscale or BGR)
            
        Returns:
            Flow field (H x W x 2) with dx, dy at each pixel
        """
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = frame1, frame2
        
        if self.flow_method == 'farneback':
            # Farneback optical flow (dense, good for general motion)
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2,
                None,
                pyr_scale=0.5,      # Image pyramid scale
                levels=3,           # Number of pyramid levels
                winsize=15,         # Averaging window size
                iterations=3,       # Iterations at each level
                poly_n=5,           # Polynomial expansion neighborhood
                poly_sigma=1.2,     # Gaussian smoothing
                flags=0
            )
        else:  # lucas_kanade
            # Lucas-Kanade sparse flow (fast, good for features)
            # Note: This requires feature points, so we'll use goodFeaturesToTrack
            corners = cv2.goodFeaturesToTrack(
                gray1,
                maxCorners=1000,
                qualityLevel=0.01,
                minDistance=10
            )
            
            if corners is None:
                # No features found, return zero flow
                return np.zeros((*gray1.shape, 2), dtype=np.float32)
            
            # Calculate flow for sparse features
            new_corners, status, _ = cv2.calcOpticalFlowPyrLK(
                gray1, gray2,
                corners,
                None,
                winSize=(15, 15),
                maxLevel=2
            )
            
            # Create dense flow field by interpolation
            # (In practice, for PVC we'd use Farneback, but this is for completeness)
            flow = np.zeros((*gray1.shape, 2), dtype=np.float32)
            for i, (old, new) in enumerate(zip(corners, new_corners)):
                if status[i]:
                    x, y = old.ravel()
                    dx, dy = (new - old).ravel()
                    flow[int(y), int(x)] = [dx, dy]
        
        return flow
    
    def extract_contour_motion(self,
                              contour: Dict,
                              flow: np.ndarray) -> Dict:
        """
        Extract motion vector for a single contour from flow field.
        
        Args:
            contour: Contour dictionary with 'points' and 'bounding_box'
            flow: Optical flow field (H x W x 2)
            
        Returns:
            Motion dictionary with average dx, dy and affine parameters
        """
        # Get contour region
        x, y, w, h = contour['bounding_box']
        
        # Clip to flow bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(flow.shape[1], x + w)
        y2 = min(flow.shape[0], y + h)
        
        if x2 <= x1 or y2 <= y1:
            # Empty region
            return {
                'translation': [0.0, 0.0],
                'rotation': 0.0,
                'scale': 1.0,
                'confidence': 0.0
            }
        
        # Extract flow vectors in contour region
        region_flow = flow[y1:y2, x1:x2]
        
        # Compute average translation
        avg_dx = float(np.mean(region_flow[:, :, 0]))
        avg_dy = float(np.mean(region_flow[:, :, 1]))
        
        # Estimate rotation and scale (simplified)
        # Full affine would require feature matching, this is an approximation
        flow_magnitude = np.sqrt(region_flow[:, :, 0]**2 + region_flow[:, :, 1]**2)
        avg_magnitude = float(np.mean(flow_magnitude))
        
        # Confidence based on flow consistency
        flow_std = float(np.std(flow_magnitude))
        confidence = 1.0 / (1.0 + flow_std) if flow_std > 0 else 1.0
        
        return {
            'translation': [avg_dx, avg_dy],
            'rotation': 0.0,  # Simplified for now
            'scale': 1.0,     # Simplified for now
            'confidence': confidence
        }
    
    def track_objects(self,
                     frames: List[np.ndarray],
                     frame_contours: List[List[Dict]]) -> List[Dict]:
        """
        Track objects across all frames.
        
        Args:
            frames: List of video frames (BGR images)
            frame_contours: List of contour lists (one per frame)
            
        Returns:
            List of tracked objects with motion trajectories
        """
        if len(frames) < 2:
            logger.warning("Need at least 2 frames for motion tracking")
            return []
        
        logger.info(f"Tracking motion across {len(frames)} frames...")
        
        # Match contours across frames (simple nearest-centroid matching)
        objects = []
        object_id = 0
        
        # Initialize objects from first frame
        for contour in frame_contours[0]:
            objects.append({
                'id': object_id,
                'contour_sequence': [contour],
                'motion_sequence': [],
                'keyframes': [0]  # Frame indices where we store full contour
            })
            object_id += 1
        
        # Process each consecutive frame pair
        for frame_idx in range(1, len(frames)):
            frame1 = frames[frame_idx - 1]
            frame2 = frames[frame_idx]
            contours_curr = frame_contours[frame_idx]
            
            # Compute optical flow
            flow = self.compute_optical_flow(frame1, frame2)
            
            # Match current contours to tracked objects
            matched = set()
            for obj in objects:
                if not obj['contour_sequence']:
                    continue
                
                last_contour = obj['contour_sequence'][-1]
                last_centroid = np.array(last_contour['centroid'])
                
                # Find nearest contour in current frame
                best_match = None
                best_distance = float('inf')
                
                for idx, contour in enumerate(contours_curr):
                    if idx in matched:
                        continue
                    
                    curr_centroid = np.array(contour['centroid'])
                    distance = np.linalg.norm(curr_centroid - last_centroid)
                    
                    # Also check area similarity
                    area_ratio = contour['area'] / (last_contour['area'] + 1e-6)
                    if area_ratio < 0.5 or area_ratio > 2.0:
                        continue  # Too different in size
                    
                    if distance < best_distance and distance < 100:  # Max tracking distance
                        best_distance = distance
                        best_match = (idx, contour)
                
                if best_match:
                    idx, contour = best_match
                    matched.add(idx)
                    
                    # Add to object trajectory
                    obj['contour_sequence'].append(contour)
                    
                    # Extract motion
                    motion = self.extract_contour_motion(last_contour, flow)
                    obj['motion_sequence'].append(motion)
                    
                    # Store keyframe?
                    if frame_idx % self.keyframe_interval == 0:
                        obj['keyframes'].append(frame_idx)
            
            # Create new objects for unmatched contours
            for idx, contour in enumerate(contours_curr):
                if idx not in matched and contour['area'] > 200:  # Significant object
                    objects.append({
                        'id': object_id,
                        'contour_sequence': [contour],
                        'motion_sequence': [],
                        'keyframes': [frame_idx]
                    })
                    object_id += 1
        
        logger.info(f"Tracked {len(objects)} objects")
        return objects
    
    def fit_motion_model(self, 
                        motion_sequence: List[Dict],
                        model_type: str = 'spline') -> Dict:
        """
        Fit a smooth motion model to a sequence of motion vectors.
        
        Args:
            motion_sequence: List of motion dictionaries
            model_type: 'spline', 'polynomial', or 'affine'
            
        Returns:
            Compact motion model parameters
        """
        if not motion_sequence:
            return {'type': 'static', 'params': []}
        
        # Extract translation vectors
        translations = np.array([m['translation'] for m in motion_sequence])
        t = np.arange(len(translations))
        
        if model_type == 'spline' and len(translations) >= 4:
            # Fit cubic spline
            try:
                tck_x = interpolate.splrep(t, translations[:, 0], s=0.5 * len(t))
                tck_y = interpolate.splrep(t, translations[:, 1], s=0.5 * len(t))
                
                return {
                    'type': 'spline',
                    'params': {
                        'knots_x': tck_x[0].tolist(),
                        'coeffs_x': tck_x[1].tolist(),
                        'degree_x': int(tck_x[2]),
                        'knots_y': tck_y[0].tolist(),
                        'coeffs_y': tck_y[1].tolist(),
                        'degree_y': int(tck_y[2])
                    }
                }
            except:
                logger.warning("Spline fitting failed, using keyframes")
        
        # Fallback: store keyframe translations
        return {
            'type': 'keyframes',
            'params': {
                'frames': t.tolist(),
                'translations': translations.tolist()
            }
        }


# Example usage
if __name__ == "__main__":
    # Test with synthetic motion
    tracker = MotionTracker()
    
    # Create two frames with a moving square
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    cv2.rectangle(frame1, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.rectangle(frame2, (120, 110), (220, 210), (255, 255, 255), -1)
    
    # Compute flow
    flow = tracker.compute_optical_flow(frame1, frame2)
    print(f"Flow shape: {flow.shape}")
    print(f"Average motion: dx={np.mean(flow[:,:,0]):.2f}, dy={np.mean(flow[:,:,1]):.2f}")

