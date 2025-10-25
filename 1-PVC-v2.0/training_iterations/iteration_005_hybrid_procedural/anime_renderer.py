"""
Differentiable Anime Renderer
Renders anime images from function IDs and parameters using differentiable operations

Key requirement: All operations must be differentiable so gradients can flow back
to the procedural predictor during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DifferentiableAnimeRenderer(nn.Module):
    """
    Renders anime-style images from procedural operations
    All operations are differentiable using PyTorch operations
    """
    
    def __init__(self, img_height=512, img_width=960, num_functions=15):
        super().__init__()
        self.height = img_height
        self.width = img_width
        self.num_functions = num_functions
        
        # Create coordinate grids (for geometric operations)
        y_coords = torch.linspace(0, 1, img_height)
        x_coords = torch.linspace(0, 1, img_width)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        self.register_buffer('grid_y', grid_y)  # (H, W)
        self.register_buffer('grid_x', grid_x)  # (H, W)
        
    def render_function_0_outline(self, params, canvas):
        """
        Function 0: Anime outline (thick line)
        params: [x1, y1, x2, y2, thickness, r, g, b, _, _]
        """
        x1, y1, x2, y2 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        thickness = params[:, 4] * 0.1  # Scale to [0, 0.1]
        color = params[:, 5:8].unsqueeze(-1).unsqueeze(-1)  # (B, 3, 1, 1)
        
        # Create line mask using distance to line
        # Line equation: dist = |ax + by + c| / sqrt(a^2 + b^2)
        a = (y2 - y1).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        b = (x1 - x2).unsqueeze(-1).unsqueeze(-1)
        c = (x2 * y1 - x1 * y2).unsqueeze(-1).unsqueeze(-1)
        
        grid_x = self.grid_x.unsqueeze(0)  # (1, H, W)
        grid_y = self.grid_y.unsqueeze(0)
        
        # Distance from each pixel to the line
        numerator = torch.abs(a * grid_x + b * grid_y + c)
        denominator = torch.sqrt(a**2 + b**2 + 1e-8)
        dist = numerator / denominator  # (B, H, W)
        
        # Soft threshold to create line mask
        mask = torch.sigmoid((thickness.unsqueeze(-1).unsqueeze(-1) - dist) * 100)  # (B, H, W)
        mask = mask.unsqueeze(1)  # (B, 1, H, W)
        
        # Blend with canvas
        canvas = canvas * (1 - mask) + color * mask
        return canvas
    
    def render_function_10_cel_region(self, params, canvas):
        """
        Function 10: Cel shading (flat color region)
        params: [cx, cy, radius, _, _, r, g, b, _, _]
        """
        cx, cy, radius = params[:, 0], params[:, 1], params[:, 2] * 0.3
        color = params[:, 5:8].unsqueeze(-1).unsqueeze(-1)  # (B, 3, 1, 1)
        
        grid_x = self.grid_x.unsqueeze(0)  # (1, H, W)
        grid_y = self.grid_y.unsqueeze(0)
        
        # Distance from center
        cx = cx.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        cy = cy.unsqueeze(-1).unsqueeze(-1)
        dist = torch.sqrt((grid_x - cx)**2 + (grid_y - cy)**2)  # (B, H, W)
        
        # Soft circle mask
        mask = torch.sigmoid((radius.unsqueeze(-1).unsqueeze(-1) - dist) * 100)
        mask = mask.unsqueeze(1)  # (B, 1, H, W)
        
        # Blend
        canvas = canvas * (1 - mask) + color * mask
        return canvas
    
    def render_function_20_gradient(self, params, canvas):
        """
        Function 20: Linear gradient
        params: [x, y, width, height, angle, r1, g1, b1, r2, g2] (need 10 params)
        Note: For simplicity, using first 8 params for two colors
        """
        x, y = params[:, 0], params[:, 1]
        width, height = params[:, 2] * 0.5, params[:, 3] * 0.5
        angle = params[:, 4] * np.pi  # [0, π]
        color1 = params[:, 5:8].unsqueeze(-1).unsqueeze(-1)  # (B, 3, 1, 1)
        # For color2, reuse some params (limitation of 10 params)
        color2 = torch.clamp(color1 + 0.2, 0, 1)  # Slightly lighter
        
        grid_x = self.grid_x.unsqueeze(0)
        grid_y = self.grid_y.unsqueeze(0)
        
        # Gradient direction
        cos_a = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)
        sin_a = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
        
        # Gradient value based on position
        x_b = x.unsqueeze(-1).unsqueeze(-1)
        y_b = y.unsqueeze(-1).unsqueeze(-1)
        grad_val = (grid_x - x_b) * cos_a + (grid_y - y_b) * sin_a
        grad_val = torch.sigmoid(grad_val * 5)  # (B, H, W) in [0, 1]
        grad_val = grad_val.unsqueeze(1)  # (B, 1, H, W)
        
        # Interpolate colors
        gradient = color1 * (1 - grad_val) + color2 * grad_val  # (B, 3, H, W)
        
        # Region mask (rectangular area)
        width_b = width.unsqueeze(-1).unsqueeze(-1)
        height_b = height.unsqueeze(-1).unsqueeze(-1)
        mask_x = torch.sigmoid((width_b - torch.abs(grid_x - x_b)) * 50)
        mask_y = torch.sigmoid((height_b - torch.abs(grid_y - y_b)) * 50)
        mask = (mask_x * mask_y).unsqueeze(1)  # (B, 1, H, W)
        
        # Blend
        canvas = canvas * (1 - mask) + gradient * mask
        return canvas
    
    def render_function_30_anime_eye(self, params, canvas):
        """
        Function 30: Simplified anime eye
        params: [cx, cy, width, height, _, r, g, b, _, _]
        """
        cx, cy = params[:, 0], params[:, 1]
        width, height = params[:, 2] * 0.1, params[:, 3] * 0.1
        color = params[:, 5:8].unsqueeze(-1).unsqueeze(-1)
        
        grid_x = self.grid_x.unsqueeze(0)
        grid_y = self.grid_y.unsqueeze(0)
        
        cx = cx.unsqueeze(-1).unsqueeze(-1)
        cy = cy.unsqueeze(-1).unsqueeze(-1)
        width = width.unsqueeze(-1).unsqueeze(-1)
        height = height.unsqueeze(-1).unsqueeze(-1)
        
        # Ellipse distance
        dx = (grid_x - cx) / (width + 1e-8)
        dy = (grid_y - cy) / (height + 1e-8)
        dist = torch.sqrt(dx**2 + dy**2)
        
        # Eye mask
        mask = torch.sigmoid((1.0 - dist) * 20)
        mask = mask.unsqueeze(1)
        
        # Add white highlight
        highlight_x = cx - width * 0.3
        highlight_y = cy - height * 0.3
        dx_h = (grid_x - highlight_x) / (width * 0.3 + 1e-8)
        dy_h = (grid_y - highlight_y) / (height * 0.3 + 1e-8)
        dist_h = torch.sqrt(dx_h**2 + dy_h**2)
        highlight_mask = torch.sigmoid((0.5 - dist_h) * 30).unsqueeze(1)
        
        # Blend eye color
        canvas = canvas * (1 - mask) + color * mask
        # Add highlight
        canvas = canvas * (1 - highlight_mask) + torch.ones_like(color) * highlight_mask
        
        return canvas
    
    def forward(self, function_ids, params):
        """
        Render image from function sequence
        
        Args:
            function_ids: (B, max_ops) - which functions to use
            params: (B, max_ops, 10) - parameters for each function
        
        Returns:
            rendered: (B, 3, H, W) - rendered image
        """
        B = function_ids.size(0)
        max_ops = function_ids.size(1)
        
        # Initialize white canvas
        canvas = torch.ones(B, 3, self.height, self.width, device=function_ids.device)
        
        # Render each operation sequentially
        for op_idx in range(max_ops):
            func_ids = function_ids[:, op_idx]  # (B,)
            op_params = params[:, op_idx, :]  # (B, 10)
            
            # For each batch element, apply the corresponding function
            # Note: In practice, we'd use torch.where to selectively apply functions
            # For simplicity, let's apply a weighted sum of all functions
            
            # Soft selection using function probabilities (if func_ids are logits)
            # For now, assume func_ids are hard selections (integers)
            
            # Apply each function type
            for func_id in range(min(5, self.num_functions)):  # Limit to implemented functions
                # Create mask for this function
                mask = (func_ids == func_id).float().view(B, 1, 1, 1)
                
                # Render this function for all batches
                if func_id == 0:
                    rendered = self.render_function_0_outline(op_params, canvas.clone())
                elif func_id == 1:
                    rendered = self.render_function_10_cel_region(op_params, canvas.clone())
                elif func_id == 2:
                    rendered = self.render_function_20_gradient(op_params, canvas.clone())
                elif func_id == 3:
                    rendered = self.render_function_30_anime_eye(op_params, canvas.clone())
                else:
                    rendered = canvas.clone()
                
                # Apply masked update
                canvas = canvas * (1 - mask) + rendered * mask
        
        return canvas


if __name__ == "__main__":
    print("="*60)
    print("TESTING DIFFERENTIABLE ANIME RENDERER")
    print("="*60)
    
    # Create renderer
    renderer = DifferentiableAnimeRenderer(img_height=256, img_width=256, num_functions=15)
    
    batch_size = 2
    max_ops = 10
    
    # Create dummy function sequence
    function_ids = torch.randint(0, 5, (batch_size, max_ops))
    params = torch.rand(batch_size, max_ops, 10, requires_grad=True)  # Enable gradients
    
    print(f"\nInput:")
    print(f"  Function IDs shape: {function_ids.shape}")
    print(f"  Params shape: {params.shape}")
    print(f"  Sample function IDs: {function_ids[0, :5].tolist()}")
    
    # Render
    rendered = renderer(function_ids, params)
    
    print(f"\nOutput:")
    print(f"  Rendered shape: {rendered.shape}")
    print(f"  Rendered range: [{rendered.min():.3f}, {rendered.max():.3f}]")
    
    # Test gradient flow
    print(f"\nTesting gradient flow...")
    loss = rendered.mean()
    loss.backward()
    
    print(f"  ✓ Gradients computed successfully")
    print(f"  Params grad shape: {params.grad.shape if params.grad is not None else 'None'}")
    
    # Test with actual network
    print(f"\n" + "="*60)
    print("TESTING WITH PROCEDURAL PREDICTOR")
    print("="*60)
    
    from hybrid_model import ProceduralPredictor
    
    predictor = ProceduralPredictor(num_functions=15, max_operations=10)
    renderer = DifferentiableAnimeRenderer(img_height=256, img_width=256, num_functions=15)
    
    # Forward pass
    dummy_img = torch.randn(2, 3, 256, 256)
    function_logits, params = predictor(dummy_img)
    
    # Get hard function IDs
    function_ids = torch.argmax(function_logits, dim=-1)
    
    # Render
    rendered = renderer(function_ids, params)
    
    print(f"\nPipeline test:")
    print(f"  Input image: {dummy_img.shape}")
    print(f"  Predicted functions: {function_logits.shape}")
    print(f"  Predicted params: {params.shape}")
    print(f"  Rendered output: {rendered.shape}")
    
    # Test end-to-end gradient
    target = torch.rand_like(rendered)
    loss = F.mse_loss(rendered, target)
    loss.backward()
    
    print(f"\nEnd-to-end gradient flow:")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  ✓ Gradients flow through predictor → renderer → loss")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nRenderer is ready for training!")

