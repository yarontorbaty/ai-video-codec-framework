"""
Contour Extractor - PVC Encoder Module

Extracts edges and contours from video frames, fits splines, and groups into objects.
This is the foundation of the procedural encoding pipeline.
"""

import cv2
import numpy as np
from scipy import interpolate
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContourExtractor:
    """
    Extracts and vectorizes contours from video frames.
    
    Pipeline:
    1. Edge detection (Canny)
    2. Contour finding
    3. Spline/polyline fitting
    4. Object grouping
    """
    
    def __init__(self,
                 canny_low: int = 50,
                 canny_high: int = 150,
                 min_contour_area: int = 100,
                 epsilon_factor: float = 0.01):
        """
        Initialize contour extractor.
        
        Args:
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            min_contour_area: Minimum area for contour to be considered
            epsilon_factor: Approximation accuracy for polyline simplification
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_contour_area = min_contour_area
        self.epsilon_factor = epsilon_factor
    
    def extract_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Extract contours from a single frame.
        
        Args:
            frame: BGR image (H x W x 3)
            
        Returns:
            List of contour dictionaries with points and properties
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            edges, 
            cv2.RETR_TREE,  # Get hierarchical structure
            cv2.CHAIN_APPROX_SIMPLE  # Compress horizontal/vertical segments
        )
        
        # Process and filter contours
        extracted_contours = []
        for idx, contour in enumerate(contours):
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            # Approximate contour with polyline
            epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            # Get hierarchy info (parent, first child, next, previous)
            h_info = hierarchy[0][idx] if hierarchy is not None else [-1, -1, -1, -1]
            
            contour_data = {
                'id': idx,
                'points': approx.reshape(-1, 2).tolist(),  # Simplified polyline
                'area': float(area),
                'perimeter': float(cv2.arcLength(contour, True)),
                'bounding_box': [int(x), int(y), int(w), int(h)],
                'centroid': [int(cx), int(cy)],
                'hierarchy': [int(h) for h in h_info],  # [parent, first_child, next, prev]
                'is_closed': True
            }
            
            extracted_contours.append(contour_data)
        
        logger.debug(f"Extracted {len(extracted_contours)} contours from frame")
        return extracted_contours
    
    def fit_spline(self, points: np.ndarray, smoothing: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a B-spline to a set of points.
        
        Args:
            points: Array of shape (N, 2) with x, y coordinates
            smoothing: Smoothing factor (0 = interpolate, >0 = approximate)
            
        Returns:
            Tuple of (spline_x, spline_y) arrays
        """
        if len(points) < 3:
            return points[:, 0], points[:, 1]
        
        # Parameterize by cumulative distance
        distances = np.concatenate([[0], np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))])
        t = distances / distances[-1]  # Normalize to [0, 1]
        
        try:
            # Fit B-spline with smoothing
            tck_x, _ = interpolate.splprep([points[:, 0]], u=t, s=smoothing * len(points))
            tck_y, _ = interpolate.splprep([points[:, 1]], u=t, s=smoothing * len(points))
            
            # Evaluate spline at original parameter values
            spline_x = interpolate.splev(t, tck_x[0])
            spline_y = interpolate.splev(t, tck_y[0])
            
            return np.array(spline_x), np.array(spline_y)
        except:
            # Fallback to original points if spline fitting fails
            logger.warning("Spline fitting failed, using polyline")
            return points[:, 0], points[:, 1]
    
    def group_contours(self, contours: List[Dict], 
                       distance_threshold: float = 50.0) -> List[List[int]]:
        """
        Group contours into objects based on spatial proximity and hierarchy.
        
        Args:
            contours: List of contour dictionaries
            distance_threshold: Maximum distance between centroids to group
            
        Returns:
            List of groups, where each group is a list of contour IDs
        """
        if not contours:
            return []
        
        # Build adjacency matrix based on distance
        n = len(contours)
        adjacency = np.zeros((n, n), dtype=bool)
        
        for i in range(n):
            for j in range(i + 1, n):
                c1, c2 = contours[i], contours[j]
                
                # Check spatial proximity
                dist = np.sqrt(
                    (c1['centroid'][0] - c2['centroid'][0])**2 +
                    (c1['centroid'][1] - c2['centroid'][1])**2
                )
                
                # Check hierarchical relationship
                is_parent_child = (
                    c1['hierarchy'][0] == j or  # c2 is parent of c1
                    c1['hierarchy'][1] == j or  # c2 is first child of c1
                    c2['hierarchy'][0] == i or  # c1 is parent of c2
                    c2['hierarchy'][1] == i     # c1 is first child of c2
                )
                
                if dist < distance_threshold or is_parent_child:
                    adjacency[i, j] = True
                    adjacency[j, i] = True
        
        # Find connected components using DFS
        visited = np.zeros(n, dtype=bool)
        groups = []
        
        def dfs(node, group):
            visited[node] = True
            group.append(node)
            for neighbor in range(n):
                if adjacency[node, neighbor] and not visited[neighbor]:
                    dfs(neighbor, group)
        
        for i in range(n):
            if not visited[i]:
                group = []
                dfs(i, group)
                groups.append(group)
        
        logger.debug(f"Grouped {n} contours into {len(groups)} objects")
        return groups
    
    def extract_video(self, video_path: str, 
                      max_frames: Optional[int] = None) -> List[List[Dict]]:
        """
        Extract contours from all frames in a video.
        
        Args:
            video_path: Path to input video file
            max_frames: Maximum number of frames to process (None = all)
            
        Returns:
            List of frame contour data (one list per frame)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_contours = []
        frame_idx = 0
        
        logger.info(f"Extracting contours from video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_idx >= max_frames:
                break
            
            contours = self.extract_frame(frame)
            frame_contours.append(contours)
            
            frame_idx += 1
            if frame_idx % 10 == 0:
                logger.info(f"Processed {frame_idx} frames...")
        
        cap.release()
        logger.info(f"Extracted contours from {frame_idx} frames")
        
        return frame_contours


# Example usage
if __name__ == "__main__":
    # Test on a sample frame
    extractor = ContourExtractor()
    
    # Create a test image with simple shapes
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.circle(test_img, (450, 200), 80, (255, 255, 255), -1)
    
    # Extract contours
    contours = extractor.extract_frame(test_img)
    print(f"Extracted {len(contours)} contours")
    
    for c in contours:
        print(f"  Contour {c['id']}: {len(c['points'])} points, area={c['area']:.1f}")

