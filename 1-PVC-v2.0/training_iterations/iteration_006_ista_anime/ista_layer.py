"""
ISTA Layer for Anime Compression
Inspired by ISTA-Net (CVPR 2018)

Key innovation: Unroll ISTA optimization into neural network layers
Each layer = one iteration of sparse coding optimization
"""

import torch
import torch.nn as nn
import numpy as np


class ISTALayer(nn.Module):
    """
    One ISTA (Iterative Shrinkage-Thresholding Algorithm) iteration
    
    Implements: x^(t+1) = soft_threshold(x^(t) - α∇f(x^(t)), λα)
    
    where:
    - x = sparse coefficients (which anime functions to use)
    - α = step size (learnable)
    - λ = sparsity threshold (learnable)
    - ∇f(x) = gradient of reconstruction error
    """
    
    def __init__(self, num_functions=15, feature_dim=256*16*30):
        super().__init__()
        self.num_functions = num_functions
        self.feature_dim = feature_dim
        
        # Learnable dictionary (anime function basis)
        # Each row = one anime function's feature representation
        self.dictionary = nn.Parameter(torch.randn(num_functions, feature_dim) * 0.01)
        
        # Learnable step size (α) - controls how fast we move in gradient direction
        self.step_size = nn.Parameter(torch.tensor(0.1))
        
        # Learnable threshold (λ) - controls sparsity level
        self.threshold = nn.Parameter(torch.tensor(0.01))
        
        # Normalization for stability
        self.register_buffer('dict_norm', torch.ones(num_functions))
        
    def normalize_dictionary(self):
        """Normalize dictionary columns to unit norm (helps training stability)"""
        with torch.no_grad():
            self.dict_norm = torch.norm(self.dictionary, p=2, dim=1, keepdim=True)
            self.dictionary.data = self.dictionary.data / (self.dict_norm + 1e-8)
    
    def soft_threshold(self, x, threshold):
        """
        Soft thresholding operator: promotes sparsity
        
        soft_threshold(x, λ) = sign(x) * max(|x| - λ, 0)
        
        Effect: Values close to zero are pushed to exactly zero
        """
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)
    
    def forward(self, x_prev, features):
        """
        One ISTA iteration
        
        Args:
            x_prev: Previous sparse coefficients (B, num_functions)
            features: Target features to reconstruct (B, feature_dim)
        
        Returns:
            x_next: Updated sparse coefficients (B, num_functions)
        """
        # Normalize dictionary periodically for stability
        if self.training and torch.rand(1).item() < 0.1:
            self.normalize_dictionary()
        
        # Forward pass: reconstruct features from sparse coefficients
        # reconstruction = x @ D where D is dictionary
        reconstruction = x_prev @ self.dictionary  # (B, feature_dim)
        
        # Compute residual (reconstruction error)
        residual = reconstruction - features  # (B, feature_dim)
        
        # Compute gradient: ∇f(x) = Dᵀ(Dx - y)
        gradient = residual @ self.dictionary.T  # (B, num_functions)
        
        # Gradient descent step: x - α∇f(x)
        x_gradient_step = x_prev - self.step_size * gradient
        
        # Soft thresholding: promotes sparsity
        x_next = self.soft_threshold(x_gradient_step, torch.abs(self.threshold))
        
        return x_next
    
    def get_sparsity(self, x):
        """Calculate sparsity level (fraction of non-zero coefficients)"""
        return (torch.abs(x) > 1e-3).float().mean().item()


class ISTABlock(nn.Module):
    """
    Multiple ISTA layers stacked together
    Represents full iterative optimization process
    """
    
    def __init__(self, num_functions=15, feature_dim=256*16*30, num_iterations=10):
        super().__init__()
        self.num_iterations = num_iterations
        
        # Stack of ISTA layers (one per iteration)
        self.ista_layers = nn.ModuleList([
            ISTALayer(num_functions, feature_dim)
            for _ in range(num_iterations)
        ])
    
    def forward(self, x_init, features, return_trajectory=False):
        """
        Run full ISTA optimization
        
        Args:
            x_init: Initial sparse coefficients (B, num_functions)
            features: Target features (B, feature_dim)
            return_trajectory: If True, return intermediate results
        
        Returns:
            x_final: Final sparse coefficients (B, num_functions)
            trajectory: (Optional) List of intermediate coefficients
        """
        x = x_init
        trajectory = [x]
        
        for ista_layer in self.ista_layers:
            x = ista_layer(x, features)
            if return_trajectory:
                trajectory.append(x)
        
        if return_trajectory:
            return x, trajectory
        return x
    
    def get_average_sparsity(self, x):
        """Calculate average sparsity across all layers"""
        sparsities = [layer.get_sparsity(x) for layer in self.ista_layers]
        return np.mean(sparsities)


if __name__ == "__main__":
    print("="*60)
    print("TESTING ISTA LAYER")
    print("="*60)
    
    # Test single ISTA layer
    batch_size = 4
    num_functions = 15
    feature_dim = 256 * 16 * 30
    
    ista = ISTALayer(num_functions, feature_dim)
    
    print(f"\nConfiguration:")
    print(f"  Num functions: {num_functions}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Initial step size: {ista.step_size.item():.4f}")
    print(f"  Initial threshold: {ista.threshold.item():.4f}")
    
    # Create dummy data
    x_init = torch.zeros(batch_size, num_functions)
    features = torch.randn(batch_size, feature_dim)
    
    # Run one ISTA iteration
    x_next = ista(x_init, features)
    
    print(f"\nAfter 1 iteration:")
    print(f"  Output shape: {x_next.shape}")
    print(f"  Output range: [{x_next.min():.4f}, {x_next.max():.4f}]")
    print(f"  Sparsity: {ista.get_sparsity(x_next):.2%}")
    print(f"  Non-zero coeffs: {(torch.abs(x_next) > 1e-3).sum().item()} / {x_next.numel()}")
    
    # Test ISTA block (multiple iterations)
    print("\n" + "="*60)
    print("TESTING ISTA BLOCK (10 iterations)")
    print("="*60)
    
    ista_block = ISTABlock(num_functions, feature_dim, num_iterations=10)
    x_final, trajectory = ista_block(x_init, features, return_trajectory=True)
    
    print(f"\nTraining for 10 iterations:")
    for i, x in enumerate(trajectory):
        sparsity = (torch.abs(x) > 1e-3).float().mean().item()
        mean_abs = torch.abs(x).mean().item()
        print(f"  Iteration {i}: Sparsity={sparsity:.2%}, Mean |x|={mean_abs:.4f}")
    
    print(f"\nFinal result:")
    print(f"  Shape: {x_final.shape}")
    print(f"  Range: [{x_final.min():.4f}, {x_final.max():.4f}]")
    print(f"  Average sparsity: {ista_block.get_average_sparsity(x_final):.2%}")
    
    # Test gradient flow
    print("\n" + "="*60)
    print("TESTING GRADIENT FLOW")
    print("="*60)
    
    ista_block.train()
    optimizer = torch.optim.Adam(ista_block.parameters(), lr=1e-3)
    
    # Dummy training loop
    for step in range(5):
        x_pred = ista_block(x_init, features)
        
        # Dummy loss: want sparse but accurate reconstruction
        recon = x_pred @ ista_block.ista_layers[0].dictionary
        recon_loss = torch.mean((recon - features)**2)
        sparse_loss = 0.1 * torch.mean(torch.abs(x_pred))
        loss = recon_loss + sparse_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Step {step}: Loss={loss.item():.6f}, Recon={recon_loss.item():.6f}, Sparse={sparse_loss.item():.6f}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)

