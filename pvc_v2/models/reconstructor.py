#!/usr/bin/env python3
"""
PVC v2.0 - Complete Reconstruction Pipeline

Maps predicted parameters to actual function calls and renders reconstructed frames.
This completes the loop: Frame → Functions → Parameters → Render → Compare
"""

import torch
import numpy as np
import cv2
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.network import PVCv2Model
from graphics.primitives import GraphicsPrimitives, FunctionCall


class PVCv2Reconstructor:
    """
    Complete reconstruction pipeline for PVC v2.0.
    
    Predicts function sequences + parameters, renders frames, measures quality.
    """
    
    # Function ID to name mapping
    FUNC_NAMES = [
        'fill_solid',             # 0
        'draw_gradient_linear',   # 1
        'draw_gradient_radial',   # 2
        'draw_rectangle',         # 3
        'draw_ellipse',          # 4
        'draw_polygon',          # 5
        'apply_gaussian_blur',   # 6
        'apply_noise',           # 7
        'blend_layers',          # 8
        'adjust_brightness',     # 9
    ]
    
    def __init__(self, model: PVCv2Model, device: torch.device):
        """
        Initialize reconstructor.
        
        Args:
            model: Trained PVC v2.0 model
            device: Torch device
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def map_parameters_to_function(self, 
                                   func_id: int,
                                   params: Dict[str, np.ndarray],
                                   frame_size: Tuple[int, int] = (256, 256)) -> Dict:
        """
        Map predicted parameters to actual function arguments.
        
        Args:
            func_id: Function ID (0-9)
            params: Predicted parameters (coords, color1, color2, scalars)
            frame_size: Frame dimensions (width, height)
            
        Returns:
            Dictionary of function arguments
        """
        width, height = frame_size
        coords = params['coords']
        color1 = tuple(params['color1'])
        color2 = tuple(params['color2'])
        scalars = params['scalars']
        
        # Denormalize coordinates to frame size
        # Coords are predicted in normalized space, map to actual pixels
        x = int(np.clip(coords[0] * width / 4, 0, width - 1))
        y = int(np.clip(coords[1] * height / 4, 0, height - 1))
        w = int(np.clip(abs(coords[2]) * width / 2, 10, width))
        h = int(np.clip(abs(coords[3]) * height / 2, 10, height))
        
        # Ensure colors are in valid range
        color1 = tuple(np.clip(color1, 0, 1))
        color2 = tuple(np.clip(color2, 0, 1))
        
        func_name = self.FUNC_NAMES[func_id]
        
        # Map based on function type
        if func_name == 'fill_solid':
            return {
                'x': x, 'y': y, 'w': w, 'h': h,
                'color': color1,
                'opacity': float(np.clip(scalars[0], 0.5, 1.0))
            }
        
        elif func_name == 'draw_gradient_linear':
            return {
                'x': x, 'y': y, 'w': w, 'h': h,
                'color1': color1,
                'color2': color2,
                'angle': float(np.clip(scalars[1] * 180, 0, 180))
            }
        
        elif func_name == 'draw_gradient_radial':
            cx = int(np.clip(coords[0] * width / 2, 0, width - 1))
            cy = int(np.clip(coords[1] * height / 2, 0, height - 1))
            radius = int(np.clip(abs(coords[2]) * max(width, height), 50, max(width, height)))
            return {
                'cx': cx, 'cy': cy, 'radius': radius,
                'color_inner': color1,
                'color_outer': color2
            }
        
        elif func_name == 'draw_rectangle':
            return {
                'x': x, 'y': y, 'w': w, 'h': h,
                'fill': color1,
                'stroke': color2 if scalars[3] > 0.5 else None,
                'stroke_width': max(1, int(abs(scalars[3]) * 3)),
                'opacity': float(np.clip(scalars[0], 0.3, 1.0))
            }
        
        elif func_name == 'draw_ellipse':
            cx = int(np.clip(coords[0] * width / 2 + width / 4, 0, width - 1))
            cy = int(np.clip(coords[1] * height / 2 + height / 4, 0, height - 1))
            rx = max(5, int(np.clip(abs(coords[2]) * width / 4, 5, width / 2)))
            ry = max(5, int(np.clip(abs(coords[3]) * height / 4, 5, height / 2)))
            return {
                'cx': cx, 'cy': cy, 'rx': rx, 'ry': ry,
                'fill': color1,
                'stroke': color2 if scalars[3] > 0.5 else None,
                'stroke_width': max(1, int(abs(scalars[3]) * 3)),
                'opacity': float(np.clip(scalars[0], 0.3, 1.0))
            }
        
        elif func_name == 'draw_polygon':
            # Generate simple polygon points
            cx = int(np.clip(coords[0] * width / 2 + width / 4, 0, width - 1))
            cy = int(np.clip(coords[1] * height / 2 + height / 4, 0, height - 1))
            radius = max(10, int(np.clip(abs(coords[2]) * width / 4, 10, width / 2)))
            num_points = max(3, int(abs(scalars[2]) * 3 + 3))  # 3-6 points
            
            points = []
            for i in range(num_points):
                angle = (2 * np.pi * i) / num_points
                px = cx + int(radius * np.cos(angle))
                py = cy + int(radius * np.sin(angle))
                points.append((px, py))
            
            return {
                'points': points,
                'fill': color1,
                'stroke': color2 if scalars[3] > 0.5 else None,
                'stroke_width': max(1, int(abs(scalars[3]) * 3)),
                'opacity': float(np.clip(scalars[0], 0.3, 1.0))
            }
        
        elif func_name == 'apply_gaussian_blur':
            return {
                'x': x, 'y': y, 'w': w, 'h': h,
                'radius': max(1, int(abs(scalars[2]) * 5))
            }
        
        elif func_name == 'apply_noise':
            return {
                'x': x, 'y': y, 'w': w, 'h': h,
                'strength': float(np.clip(abs(scalars[0]) * 0.2, 0.01, 0.2)),
                'seed': int(abs(scalars[1]) * 10000)
            }
        
        elif func_name == 'adjust_brightness':
            return {
                'x': x, 'y': y, 'w': w, 'h': h,
                'factor': float(np.clip(scalars[0] * 0.5 + 1.0, 0.5, 1.5))
            }
        
        else:
            # Default fallback
            return {
                'x': x, 'y': y, 'w': w, 'h': h,
                'color': color1
            }
    
    def reconstruct_frame(self, original_frame: np.ndarray) -> Tuple[np.ndarray, List[FunctionCall], int]:
        """
        Reconstruct a frame using PVC v2.0.
        
        Args:
            original_frame: Original frame (H x W x 3), uint8
            
        Returns:
            (reconstructed_frame, function_calls, total_bytes)
        """
        # Resize to model input size
        frame_resized = cv2.resize(original_frame, (256, 256))
        
        with torch.no_grad():
            # Prepare input
            frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
            
            # Encode
            features = self.model.encoder(frame_tensor)
            
            # Decode sequence
            _, predicted_sequence = self.model.decoder(features)
            
            # Predict parameters for each function and render
            renderer = GraphicsPrimitives(256, 256)
            renderer.create_canvas()
            
            function_calls = []
            total_bytes = 0
            
            for func_id_tensor in predicted_sequence[0]:
                func_id = func_id_tensor.item()
                
                # Stop at END token
                if func_id >= self.model.num_functions:
                    break
                
                # Predict parameters
                param_dict = self.model.param_predictor(features, func_id_tensor.unsqueeze(0))
                
                # Convert to numpy
                params = {
                    'coords': param_dict['coords'][0].cpu().numpy(),
                    'color1': param_dict['color1'][0].cpu().numpy(),
                    'color2': param_dict['color2'][0].cpu().numpy(),
                    'scalars': param_dict['scalars'][0].cpu().numpy(),
                }
                
                # Map to function arguments
                func_args = self.map_parameters_to_function(func_id, params)
                func_name = self.FUNC_NAMES[func_id]
                
                # Execute function
                try:
                    func = getattr(renderer, func_name)
                    call = func(**func_args)
                    function_calls.append(call)
                    total_bytes += call.estimate_size()
                except Exception as e:
                    print(f"Warning: Failed to execute {func_name}: {e}")
            
            # Get rendered frame
            reconstructed = renderer.get_canvas()
        
        return reconstructed, function_calls, total_bytes
    
    def evaluate_reconstruction(self,
                               original: np.ndarray,
                               reconstructed: np.ndarray) -> Dict[str, float]:
        """
        Evaluate reconstruction quality.
        
        Args:
            original: Original frame (H x W x 3), uint8
            reconstructed: Reconstructed frame (H x W x 3), uint8
            
        Returns:
            Dictionary of metrics
        """
        # Convert to float [0, 1]
        orig_float = original.astype(np.float32) / 255.0
        recon_float = reconstructed.astype(np.float32) / 255.0
        
        # Ensure same size
        if orig_float.shape != recon_float.shape:
            recon_float = cv2.resize(recon_float, (orig_float.shape[1], orig_float.shape[0]))
        
        # Calculate metrics
        psnr_val = psnr(orig_float, recon_float, data_range=1.0)
        ssim_val = ssim(orig_float, recon_float, data_range=1.0, channel_axis=2)
        
        # Calculate MSE
        mse = np.mean((orig_float - recon_float) ** 2)
        
        return {
            'psnr': psnr_val,
            'ssim': ssim_val,
            'mse': mse
        }


if __name__ == '__main__':
    """Test reconstruction pipeline."""
    print("="*60)
    print("PVC v2.0 - Complete Reconstruction Test")
    print("="*60)
    print()
    
    # Load model
    print("🧠 Loading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PVCv2Model(
        feature_dim=256,
        hidden_dim=256,
        num_functions=10,
        max_sequence_length=20
    ).to(device)
    model.load_state_dict(torch.load("/tmp/pvc_v2_poc_model.pth", map_location=device))
    print(f"   Device: {device}")
    print()
    
    # Create reconstructor
    reconstructor = PVCv2Reconstructor(model, device)
    
    # Test on synthetic sample
    print("📊 Testing on synthetic sample...")
    from training.synthetic_generator import SyntheticDataGenerator
    generator = SyntheticDataGenerator(256, 256)
    original, ground_truth_funcs = generator.generate_anime_like_scene()
    
    # Reconstruct
    reconstructed, predicted_funcs, total_bytes = reconstructor.reconstruct_frame(original)
    
    # Evaluate
    metrics = reconstructor.evaluate_reconstruction(original, reconstructed)
    
    print(f"\n📊 Results:")
    print(f"   Ground truth functions: {len(ground_truth_funcs)}")
    print(f"   Predicted functions:    {len(predicted_funcs)}")
    print(f"   Compressed size:        {total_bytes} bytes")
    print(f"   Original size:          {256*256*3} bytes")
    print(f"   Compression:            {(1 - total_bytes/(256*256*3))*100:.1f}%")
    print(f"\n   PSNR: {metrics['psnr']:.2f} dB")
    print(f"   SSIM: {metrics['ssim']:.4f}")
    print(f"   MSE:  {metrics['mse']:.6f}")
    print()
    
    # Save comparison
    comparison = np.hstack([original, reconstructed])
    cv2.imwrite("/tmp/pvc_v2_reconstruction_test.png", comparison)
    print(f"✅ Saved comparison: /tmp/pvc_v2_reconstruction_test.png")
    print()
    
    print("="*60)
    print("✅ Reconstruction Pipeline Complete!")
    print("="*60)

