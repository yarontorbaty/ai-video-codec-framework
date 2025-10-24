# Iteration 006: ISTA-Net Inspired Anime Codec

**Status:** Planning Phase  
**Date:** October 24, 2025  
**Inspiration:** ISTA-Net (CVPR 2018) - Interpretable Optimization-Inspired Deep Network

---

## 🎯 Goal

Create an **optimization-inspired neural codec** that:
1. Learns **sparse anime basis functions** (interpretable)
2. Uses **iterative refinement** (ISTA-style optimization)
3. Achieves **10-15 KB per frame** with better convergence than pure neural

---

## 🧠 ISTA Background

**ISTA (Iterative Shrinkage-Thresholding Algorithm):**

Solves the optimization problem:
```
min_x ||y - Φx||² + λ||x||₁

where:
- y = observed image
- Φ = dictionary/basis (our anime functions)
- x = sparse coefficients (which functions to use)
- λ = sparsity penalty
```

**ISTA Update Rule:**
```
x^(t+1) = soft_threshold(x^(t) - α∇f(x^(t)), λα)

where:
- ∇f(x) = Φᵀ(Φx - y)  (gradient)
- soft_threshold = sign(x) * max(|x| - threshold, 0)
```

**ISTA-Net Innovation:**
- Unroll ISTA iterations into neural network layers
- Make Φ, λ, α **learnable parameters**
- Each layer = one ISTA iteration (interpretable!)

---

## 🏗️ Architecture for Anime Compression

### **ISTA-Net Anime Codec Architecture:**

```
Input Image (512×960)
    ↓
┌─────────────────────────────────────────────────────┐
│ Stage 1: Feature Extraction (CNN Encoder)           │
│  - Extract high-level features                      │
│  - Output: Feature map (256×16×30)                  │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Stage 2: ISTA Unrolling (N=10 layers)               │
│  Layer 1: x₁ = SoftThreshold(x₀ - α₁∇f(x₀), λ₁)    │
│  Layer 2: x₂ = SoftThreshold(x₁ - α₂∇f(x₁), λ₂)    │
│  ...                                                 │
│  Layer N: xₙ = SoftThreshold(xₙ₋₁ - αₙ∇f(xₙ₋₁), λₙ)│
│                                                      │
│  Output: Sparse coefficients (15 anime functions)   │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Stage 3: Anime Dictionary Rendering                 │
│  - Multiply sparse coeffs by anime function basis   │
│  - Procedural rendering (differentiable)            │
│  - Output: Procedural base image                    │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Stage 4: Residual Refinement (Lightweight AE)       │
│  - Compress residual = input - procedural_base      │
│  - Output: Final image                              │
└─────────────────────────────────────────────────────┘
```

---

## 📐 Model Components

### **1. Feature Extractor**

```python
class FeatureExtractor(nn.Module):
    """Extract features for ISTA initialization"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 30))
        )
    
    def forward(self, x):
        return self.encoder(x)  # (B, 256, 16, 30)
```

### **2. ISTA Layer (Unrolled)**

```python
class ISTALayer(nn.Module):
    """One ISTA iteration as a neural network layer"""
    def __init__(self, num_functions=15, feature_dim=256):
        super().__init__()
        
        # Learnable dictionary (anime basis functions)
        self.dictionary = nn.Parameter(torch.randn(num_functions, feature_dim))
        
        # Learnable step size (α)
        self.step_size = nn.Parameter(torch.tensor(0.1))
        
        # Learnable threshold (λ)
        self.threshold = nn.Parameter(torch.tensor(0.01))
    
    def soft_threshold(self, x, threshold):
        """Soft thresholding (promotes sparsity)"""
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)
    
    def forward(self, x_prev, features):
        """
        Args:
            x_prev: Previous coefficients (B, num_functions)
            features: Input features (B, feature_dim)
        Returns:
            x_next: Updated coefficients (B, num_functions)
        """
        # Gradient step: x - α∇f(x)
        # ∇f(x) = Dᵀ(Dx - y) where D=dictionary, y=features
        reconstruction = x_prev @ self.dictionary  # (B, feature_dim)
        residual = reconstruction - features  # (B, feature_dim)
        gradient = residual @ self.dictionary.T  # (B, num_functions)
        
        x_gradient_step = x_prev - self.step_size * gradient
        
        # Soft thresholding (sparsity)
        x_next = self.soft_threshold(x_gradient_step, self.threshold)
        
        return x_next
```

### **3. ISTA-Net (Full Network)**

```python
class ISTANetAnimeCodec(nn.Module):
    """Complete ISTA-Net for anime compression"""
    def __init__(self, num_functions=15, num_ista_layers=10):
        super().__init__()
        self.num_functions = num_functions
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor()
        
        # Initial coefficient predictor
        self.init_coeff = nn.Linear(256 * 16 * 30, num_functions)
        
        # ISTA layers (unrolled optimization)
        self.ista_layers = nn.ModuleList([
            ISTALayer(num_functions, 256 * 16 * 30)
            for _ in range(num_ista_layers)
        ])
        
        # Residual autoencoder (same as iteration_005)
        self.residual_encoder = ResidualEncoder(latent_channels=32)
        self.residual_decoder = ResidualDecoder(latent_channels=32)
    
    def forward(self, img, anime_renderer):
        B = img.size(0)
        
        # Extract features
        features = self.feature_extractor(img)  # (B, 256, 16, 30)
        features_flat = features.view(B, -1)  # (B, 256*16*30)
        
        # Initialize sparse coefficients
        x = torch.zeros(B, self.num_functions, device=img.device)
        
        # ISTA iterations (sparse optimization)
        for ista_layer in self.ista_layers:
            x = ista_layer(x, features_flat)
        
        # Normalize coefficients to [0, 1] (function parameters)
        coeffs = torch.sigmoid(x)  # (B, num_functions)
        
        # Render with anime functions
        procedural_base = anime_renderer(coeffs)  # (B, 3, H, W)
        
        # Compute and compress residual
        residual = img - procedural_base
        residual_latent = self.residual_encoder(residual)
        residual_reconstructed = self.residual_decoder(residual_latent)
        
        # Final output
        final = torch.clamp(procedural_base + residual_reconstructed, 0, 1)
        
        return final, procedural_base, coeffs, residual_latent
```

---

## 📊 Key Advantages Over Iteration 005

| Feature | Iter 005 (Transformer) | **Iter 006 (ISTA-Net)** |
|---------|------------------------|-------------------------|
| **Interpretability** | Medium (attention weights) | **Very High** (optimization path) |
| **Sparsity** | No guarantee | **Built-in** (soft thresholding) |
| **Parameters** | 11M (transformer) | **~2M** (much smaller!) |
| **Training stability** | Moderate | **Better** (optimization-inspired) |
| **Inference speed** | Slower (transformer) | **Faster** (10 matrix ops) |
| **File size** | ~16 KB | **~12 KB** (sparser) |
| **Convergence** | Moderate | **Faster** (structured) |

---

## 🎓 Training Strategy

### **Loss Function:**

```python
# 1. Reconstruction loss
recon_loss = MSE(final_output, target)

# 2. Sparsity penalty (L1 on coefficients)
sparsity_loss = lambda_sparse * torch.sum(torch.abs(coeffs))

# 3. Dictionary orthogonality (prevent redundancy)
dict_matrix = torch.stack([layer.dictionary for layer in ista_layers])
ortho_loss = lambda_ortho * torch.sum((dict_matrix @ dict_matrix.T - I)**2)

# Total loss
loss = recon_loss + sparsity_loss + ortho_loss
```

### **Training Phases:**

**Phase 1: Pre-train ISTA-Net (2-3 days)**
- Freeze residual autoencoder
- Train only ISTA layers to learn sparse anime basis
- Expected PSNR: 15-20 dB

**Phase 2: Train Residual (1 day)**
- Freeze ISTA-Net
- Train residual autoencoder
- Expected PSNR: 28-32 dB

**Phase 3: Joint Fine-tuning (1 day)**
- Train everything end-to-end
- Expected PSNR: 30-35 dB

---

## 🔬 Expected Results

### **File Size Breakdown:**

| Component | Size | Notes |
|-----------|------|-------|
| **Sparse coefficients** | 15 × 2 bytes = 30 bytes | Float16, very sparse |
| **Function parameters** | 15 × 10 × 2 bytes = 300 bytes | Per-function params |
| **Procedural total** | **~0.3 KB** | Ultra-compact! |
| **Residual latent** | 32×16×30 float32 = 60 KB | Before compression |
| **Residual compressed** | **~12 KB** | INT8 + GZIP |
| **Total** | **~12.3 KB** | **Best yet!** |

### **PSNR Expectations:**

- Epoch 10: 18-22 dB
- Epoch 25: 25-28 dB
- Epoch 50: 30-33 dB
- Epoch 100: 32-36 dB (better than iter 005!)

---

## 🚀 Implementation Plan

### **Step 1: Build ISTA Layers (Day 1)**
- Implement `ISTALayer` with learnable dictionary
- Implement soft thresholding
- Test gradient flow

### **Step 2: Create Anime Dictionary Renderer (Day 1)**
- Differentiable renderer for 15 anime functions
- Test backward pass through renderer
- Verify gradients

### **Step 3: Train ISTA-Net (Days 2-3)**
- Pre-train on anime dataset
- Monitor sparsity and reconstruction
- Analyze learned dictionary

### **Step 4: Add Residual + Fine-tune (Days 4-5)**
- Integrate residual autoencoder
- Joint training
- Evaluate final results

---

## 📝 Files to Create

1. `ista_layer.py` - ISTA layer implementation
2. `ista_codec.py` - Full ISTA-Net codec
3. `anime_renderer_differentiable.py` - Differentiable anime rendering
4. `train_ista_codec.py` - Training script
5. `README.md` - Documentation

---

## 🎯 Success Criteria

**Minimum viable:**
- ✅ 12-15 KB per frame
- ✅ 28-30 dB PSNR
- ✅ Sparse coefficients (< 5 non-zero per frame)

**Stretch goals:**
- 🎯 10 KB per frame
- 🎯 32-35 dB PSNR
- 🎯 Interpretable learned dictionary
- 🎯 Faster than iteration 005

---

**Ready to implement once AMI is available!** 🚀

---

**Last Updated:** October 24, 2025  
**Status:** Design complete, awaiting infrastructure

