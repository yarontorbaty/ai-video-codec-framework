# PVC v2.0 SOTA Hybrid Codec - Architecture Design

**Goal:** Achieve 35-45 dB PSNR (production quality, match/beat HEVC)

**Date:** October 19, 2025  
**Current:** 19.91 dB  
**Target:** 35-45 dB  
**Gap:** +15-25 dB needed

---

## 🎯 Strategy Overview

To jump from 19.91 dB to 35-45 dB, we need **state-of-the-art techniques**:

1. **Enhanced Architecture** - U-Net with skip connections
2. **Perceptual Loss** - VGG-based feature matching
3. **Attention Mechanisms** - Focus on important regions
4. **Multi-Scale Processing** - Pyramidal residuals
5. **Advanced Training** - Progressive, adversarial, more data

---

## 🏗️ Enhanced Architecture

### Current (Simple) vs SOTA (Advanced)

```
┌────────────────────────────────────────────────────────────────┐
│                    CURRENT (19.91 dB)                          │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Encoder: 3→32→16→8 (3 layers)                                │
│  DCT: FFT-based approximation                                  │
│  Decoder: 8→16→32→3 (3 layers)                                │
│                                                                 │
│  Issues:                                                       │
│  ❌ Too shallow (limited capacity)                             │
│  ❌ No skip connections (lost details)                         │
│  ❌ Approximated DCT (quality loss)                            │
│  ❌ Simple MSE loss (not perceptual)                           │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                    SOTA (35-45 dB) - NEW!                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Encoder: U-Net style (8 layers + skip connections)           │
│     3→64→128→256→512 (downsampling)                           │
│     ↓                                                          │
│  Bottleneck: 512→512 with attention                           │
│     ↓                                                          │
│  Decoder: 512→256→128→64→3 (upsampling)                       │
│     ↑ skip ↑ skip ↑ skip ↑ skip                               │
│                                                                 │
│  Advanced Features:                                            │
│  ✅ Deep network (high capacity)                               │
│  ✅ Skip connections (preserve details)                        │
│  ✅ Attention (focus on important regions)                     │
│  ✅ Multi-scale (pyramidal processing)                         │
│  ✅ Perceptual loss (VGG features)                             │
│  ✅ Proper DCT implementation                                  │
└────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Component Design

### 1. Enhanced Residual Encoder (U-Net Style)

```python
class SOTAResidualEncoder(nn.Module):
    """
    State-of-the-art residual encoder with U-Net architecture.
    
    Architecture:
    - Encoder path: 5 downsampling blocks
    - Bottleneck: Attention module
    - Skip connections at each level
    - Multi-scale feature extraction
    
    Input: Residual (B, 3, 256, 256)
    Output: Multi-scale compressed features
    """
    
    def __init__(self, base_channels=64):
        # Encoder blocks (downsampling)
        self.enc1 = ConvBlock(3, 64)      # 256×256
        self.enc2 = ConvBlock(64, 128)    # 128×128
        self.enc3 = ConvBlock(128, 256)   # 64×64
        self.enc4 = ConvBlock(256, 512)   # 32×32
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            AttentionBlock(512),
            ConvBlock(512, 512)
        )  # 32×32
        
        # Multi-scale DCT compression
        self.dct_compress = MultiScaleDCT(
            scales=[8, 16, 32],
            quality_factors=[15, 25, 35]
        )
```

### 2. Attention Module

```python
class AttentionBlock(nn.Module):
    """
    Spatial attention to focus on important regions.
    
    Computes attention weights based on feature importance,
    allowing the network to focus on areas that need refinement.
    """
    
    def __init__(self, channels):
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # Compute attention map
        Q = self.query(x)  # (B, C//8, H, W)
        K = self.key(x)    # (B, C//8, H, W)
        V = self.value(x)  # (B, C, H, W)
        
        # Attention weights
        attention = F.softmax(
            torch.bmm(Q.flatten(2), K.flatten(2).transpose(1, 2)),
            dim=-1
        )
        
        # Apply attention
        out = torch.bmm(V.flatten(2), attention.transpose(1, 2))
        out = out.view_as(x)
        
        # Residual connection with learnable weight
        return x + self.gamma * out
```

### 3. Multi-Scale DCT Compression

```python
class MultiScaleDCT(nn.Module):
    """
    Multi-scale DCT compression for pyramidal processing.
    
    Applies DCT at multiple scales:
    - 8×8 blocks: Fine details (Q=15, aggressive)
    - 16×16 blocks: Medium details (Q=25, moderate)
    - 32×32 blocks: Coarse details (Q=35, conservative)
    
    This allows efficient compression while preserving
    details at multiple frequency ranges.
    """
    
    def __init__(self, scales=[8, 16, 32], quality_factors=[15, 25, 35]):
        self.scales = scales
        self.quality_factors = quality_factors
        
        # Proper DCT matrices (not FFT approximation)
        self.dct_matrices = self._init_dct_matrices()
    
    def forward(self, x):
        compressed = []
        
        for scale, Q in zip(self.scales, self.quality_factors):
            # Apply proper DCT at this scale
            dct_coeffs = self.apply_dct(x, block_size=scale)
            
            # Quantize with scale-specific quality
            quantized = self.quantize(dct_coeffs, Q)
            
            compressed.append(quantized)
        
        return compressed
```

### 4. Enhanced Residual Decoder (U-Net Style)

```python
class SOTAResidualDecoder(nn.Module):
    """
    State-of-the-art residual decoder with U-Net architecture.
    
    Architecture:
    - Decoder path: 5 upsampling blocks
    - Skip connections from encoder
    - Multi-scale reconstruction
    - Progressive refinement
    
    Input: Multi-scale compressed features + skip connections
    Output: Reconstructed residual (B, 3, 256, 256)
    """
    
    def __init__(self):
        # Multi-scale IDCT decompression
        self.idct_decompress = MultiScaleIDCT(
            scales=[8, 16, 32]
        )
        
        # Decoder blocks (upsampling) with skip connections
        self.dec4 = ConvBlock(512 + 512, 256)  # +512 from skip
        self.dec3 = ConvBlock(256 + 256, 128)  # +256 from skip
        self.dec2 = ConvBlock(128 + 128, 64)   # +128 from skip
        self.dec1 = ConvBlock(64 + 64, 32)     # +64 from skip
        
        # Final refinement
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )
```

---

## 📊 Loss Functions

### Combined Loss (Multiple Objectives)

```python
class SOTACombinedLoss(nn.Module):
    """
    State-of-the-art combined loss for residual training.
    
    Components:
    1. Pixel Loss (MSE): Ensure pixel-level accuracy
    2. Perceptual Loss (VGG): Match high-level features
    3. Style Loss: Match texture statistics
    4. Size Penalty: Encourage compression
    
    Weights: 0.2 pixel + 0.5 perceptual + 0.2 style + 0.1 size
    """
    
    def __init__(self):
        self.mse = nn.MSELoss()
        self.perceptual = VGGPerceptualLoss()
        self.style = StyleLoss()
    
    def forward(self, pred, target, compressed):
        # 1. Pixel-level loss
        pixel_loss = self.mse(pred, target)
        
        # 2. Perceptual loss (VGG features)
        perceptual_loss = self.perceptual(pred, target)
        
        # 3. Style loss (Gram matrices)
        style_loss = self.style(pred, target)
        
        # 4. Size penalty (encourage sparsity)
        size_penalty = sum(
            torch.mean(torch.abs(c)) 
            for c in compressed
        )
        
        # Combined
        total = (
            0.2 * pixel_loss +
            0.5 * perceptual_loss +
            0.2 * style_loss +
            0.1 * size_penalty
        )
        
        return total, {
            'pixel': pixel_loss.item(),
            'perceptual': perceptual_loss.item(),
            'style': style_loss.item(),
            'size': size_penalty.item()
        }
```

---

## 🎓 Advanced Training Strategy

### Progressive Training (3 Phases)

```python
# Phase 1: Coarse training (5 epochs)
# - Low resolution (128×128)
# - Simple MSE loss
# - Learn basic structure
# Target: 15-20 dB

# Phase 2: Fine training (15 epochs)
# - Full resolution (256×256)
# - Add perceptual loss
# - Refine details
# Target: 25-30 dB

# Phase 3: Refinement (30 epochs)
# - Full resolution
# - Add style loss
# - Polish quality
# Target: 35-45 dB
```

### Data Augmentation

```python
augmentation = A.Compose([
    A.RandomCrop(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
])
```

### Learning Rate Schedule

```python
# Cosine annealing with warm restarts
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,    # First restart after 10 epochs
    T_mult=2,  # Double period each restart
    eta_min=1e-6
)
```

---

## 📈 Expected Performance

### Quality Progression

| Phase | Epochs | PSNR | SSIM | Time |
|-------|--------|------|------|------|
| Current | 0 | 19.91 dB | 0.72 | - |
| Phase 1 (Coarse) | 5 | 22-25 dB | 0.80 | 30 min |
| Phase 2 (Fine) | 20 | 28-32 dB | 0.88 | 2 hrs |
| Phase 3 (Refine) | 50 | **35-45 dB** | **0.92+** | 5 hrs |

### Resource Requirements

| Resource | Current | SOTA |
|----------|---------|------|
| Model Size | 68 KB | ~2 MB |
| Parameters | 67K | ~10M |
| GPU Memory | 2 GB | 8 GB |
| Training Time | 27 min | 5-8 hrs |
| Inference Speed | Fast | Medium |

---

## 🛠️ Implementation Plan

### Phase 1: Architecture (2-3 hours)
1. Implement U-Net encoder/decoder
2. Add attention modules
3. Multi-scale DCT processing
4. Proper DCT (not FFT)
5. Test forward/backward pass

### Phase 2: Loss Functions (1-2 hours)
1. Implement perceptual loss (VGG)
2. Implement style loss (Gram matrices)
3. Combined loss with weights
4. Test on dummy data

### Phase 3: Training (5-8 hours)
1. Generate large training set (50K samples)
2. Progressive training (3 phases)
3. Data augmentation
4. Learning rate scheduling
5. Monitor convergence

### Phase 4: Evaluation (1 hour)
1. Test on held-out set
2. Measure PSNR/SSIM
3. Visual comparison
4. Compression analysis

---

## 💡 Key Innovations

### 1. U-Net with Skip Connections
**Why:** Preserves fine details lost in downsampling  
**Impact:** +5-8 dB expected

### 2. Attention Mechanisms
**Why:** Focus network capacity on important regions  
**Impact:** +2-4 dB expected

### 3. Multi-Scale Processing
**Why:** Different frequencies need different treatment  
**Impact:** +3-5 dB expected

### 4. Perceptual + Style Loss
**Why:** Optimize for human perception, not just pixels  
**Impact:** +5-10 dB expected (especially perceptual quality)

### 5. Progressive Training
**Why:** Stable convergence, better final quality  
**Impact:** +2-3 dB expected

**Total Expected:** +15-25 dB (to reach 35-45 dB) ✅

---

## 🎯 Success Criteria

### Must Have:
- ✅ PSNR: 35-45 dB
- ✅ SSIM: 0.90-0.95
- ✅ Compression: 85-92%
- ✅ Match/beat HEVC quality

### Nice to Have:
- Better than HEVC compression
- Faster than HEVC encoding
- Novel approach (procedural + neural)
- Publication-worthy results

---

## 🚀 Let's Build It!

**Next Immediate Steps:**
1. Implement U-Net encoder with attention
2. Implement U-Net decoder with skip connections
3. Implement multi-scale DCT
4. Implement combined loss (perceptual + style)
5. Train progressively (coarse → fine → refine)
6. Evaluate and measure 35-45 dB

**Estimated Total Time:** 8-12 hours

**Ready to start coding!** 🎨

This will be a **production-quality, publication-worthy codec**! 🏆

