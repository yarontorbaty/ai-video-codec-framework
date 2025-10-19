# PVC v2.0 Hybrid Architecture Design

**Goal:** Achieve 30-40 dB PSNR by combining procedural structure with neural residuals

**Date:** October 19, 2025  
**Estimated Time:** 6-8 hours  
**Target:** Production-quality codec

---

## 🎯 Architecture Overview

### Two-Stage Hybrid Codec

```
┌─────────────────────────────────────────────────────────────┐
│                     ENCODER PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input Frame (256x256x3)                                    │
│         │                                                    │
│         ├──► Stage 1: Procedural Encoder                    │
│         │    └─► PVC v2.0 Model (existing)                  │
│         │        └─► Function IDs + Parameters               │
│         │            └─► Reconstruct coarse frame           │
│         │                └─► ~11 dB PSNR                    │
│         │                                                    │
│         └──► Stage 2: Residual Encoder (NEW)                │
│              ├─► Input: Original - Coarse                   │
│              ├─► Architecture: Lightweight CNN              │
│              ├─► Compression: DCT/Quantization              │
│              └─► Output: Compressed residuals (5-10%)       │
│                                                              │
│  Output: {functions, params, residuals}                     │
│          Size: 95-98% compression                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     DECODER PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: {functions, params, residuals}                      │
│         │                                                    │
│         ├──► Stage 1: Procedural Decoder                    │
│         │    └─► Execute functions with params              │
│         │        └─► Coarse reconstruction (~11 dB)         │
│         │                                                    │
│         └──► Stage 2: Residual Decoder (NEW)                │
│              ├─► Decompress residuals                       │
│              ├─► Architecture: Lightweight CNN              │
│              └─► Add to coarse frame                        │
│                                                              │
│  Output: Final Frame                                        │
│          Quality: 30-40 dB PSNR                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Component Design

### 1. Procedural Stage (Existing - 95% compression)
**Already implemented:**
- PVC v2.0 model with parameter supervision
- Achieves 11.22 dB PSNR
- Compresses to ~5% of original size
- Provides coarse structure/layout

**No changes needed** - use best existing model

### 2. Residual Encoder (NEW)
```python
class ResidualEncoder(nn.Module):
    """
    Lightweight encoder for residual details.
    
    Input: Residual image (original - coarse)
    Output: Compressed representation
    
    Architecture:
    - 3 conv layers (channel reduction)
    - DCT transformation
    - Quantization (adjustable quality)
    - Entropy coding
    
    Compression: 5-10% additional data
    """
```

**Key features:**
- Input: 256×256×3 residual image
- CNN: 3→32→16→8 channels
- DCT: 8×8 blocks
- Quantization: Quality parameter Q (10-50)
- Output: Compressed residuals

### 3. Residual Decoder (NEW)
```python
class ResidualDecoder(nn.Module):
    """
    Lightweight decoder for residual reconstruction.
    
    Input: Compressed residuals
    Output: Residual image
    
    Architecture:
    - Entropy decoding
    - Dequantization
    - Inverse DCT
    - 3 transposed conv layers
    
    Output: 256×256×3 residual
    """
```

**Key features:**
- Inverse of encoder operations
- Skip connections for quality
- Output: Full-resolution residuals
- Add to coarse reconstruction

---

## 📊 Expected Performance

### Compression Breakdown

| Stage | Size | PSNR | Purpose |
|-------|------|------|---------|
| Procedural | 95% reduction | 11 dB | Structure/layout |
| Residuals | 5-10% additional | +20-30 dB | Fine details |
| **Total** | **90-95% compression** | **30-40 dB** | **Production quality** |

### Quality Comparison

| Codec | Bitrate | PSNR | Compression |
|-------|---------|------|-------------|
| Source | 100% | ∞ | 0% |
| HEVC (reference) | 10 Mbps | 38 dB | 95% |
| **Hybrid PVC** | **5-10 Mbps** | **30-40 dB** | **90-95%** ✅ |
| PVC alone | 5 Mbps | 11 dB | 95% |

---

## 🔧 Implementation Plan

### Phase 1: Residual Encoder/Decoder (2-3 hours)

**Files to create:**
1. `pvc_v2/models/residual_encoder.py` - CNN + DCT compression
2. `pvc_v2/models/residual_decoder.py` - Inverse operations
3. `pvc_v2/models/hybrid_codec.py` - Combined pipeline

**Architecture details:**
```python
# Residual Encoder
Conv2d(3, 32, 3, padding=1) + ReLU
Conv2d(32, 16, 3, padding=1) + ReLU  
Conv2d(16, 8, 3, padding=1)
DCT transform (8x8 blocks)
Quantization (Q parameter)

# Residual Decoder
Dequantization
Inverse DCT
ConvTranspose2d(8, 16, 3, padding=1) + ReLU
ConvTranspose2d(16, 32, 3, padding=1) + ReLU
ConvTranspose2d(32, 3, 3, padding=1)
```

### Phase 2: Integration (1 hour)

**Create hybrid codec:**
```python
class HybridPVCCodec:
    def __init__(self):
        self.procedural = load_best_pvc_model()  # 11.22 dB
        self.residual_encoder = ResidualEncoder()
        self.residual_decoder = ResidualDecoder()
    
    def encode(self, frame):
        # Stage 1: Procedural
        funcs, params = self.procedural.encode(frame)
        coarse = self.procedural.decode(funcs, params)
        
        # Stage 2: Residual
        residual = frame - coarse
        compressed_residual = self.residual_encoder(residual)
        
        return {
            'functions': funcs,
            'params': params,
            'residuals': compressed_residual
        }
    
    def decode(self, data):
        # Stage 1: Procedural
        coarse = self.procedural.decode(
            data['functions'], 
            data['params']
        )
        
        # Stage 2: Residual
        residual = self.residual_decoder(data['residuals'])
        
        # Combine
        return coarse + residual
```

### Phase 3: Training (2-3 hours)

**Training strategy:**
1. Use existing PVC model (frozen)
2. Train only residual encoder/decoder
3. Loss: MSE on residuals + size penalty
4. Optimize Q parameter for compression/quality tradeoff

**Training script:**
```python
# Train residual codec
for epoch in epochs:
    for frame in dataset:
        # Get coarse reconstruction (frozen PVC)
        with torch.no_grad():
            funcs, params = pvc_model(frame)
            coarse = reconstruct(funcs, params)
        
        # Train residual codec
        residual = frame - coarse
        compressed = residual_encoder(residual)
        reconstructed_residual = residual_decoder(compressed)
        
        # Loss: reconstruction + size penalty
        loss = mse_loss(reconstructed_residual, residual)
        loss += lambda * size_penalty(compressed)
        
        loss.backward()
        optimizer.step()
```

### Phase 4: Evaluation (30 min)

**Metrics to measure:**
- PSNR: Target 30-40 dB ✅
- SSIM: Target 0.85-0.95 ✅
- Compression: Target 90-95% ✅
- Bitrate: Target 5-10 Mbps ✅

---

## 🎯 Success Criteria

### Must Have:
- ✅ PSNR: 30-40 dB (vs 11.22 dB baseline)
- ✅ Compression: 90-95% (maintain or better)
- ✅ Combined size: < 10% of original

### Nice to Have:
- SSIM: 0.85-0.95
- Faster than HEVC encoding
- Better compression than H.264

---

## 💡 Why This Will Work

### 1. Proven Approach
- JPEG uses similar DCT compression
- H.264/HEVC use residuals extensively
- Combining procedural + residuals is novel but grounded

### 2. Best of Both Worlds
- **Procedural:** Excellent compression for structure (95%)
- **Residuals:** Proven method for fine details (5-10%)
- **Combined:** Production-quality codec

### 3. Computational Efficiency
- Procedural stage: Fast (already optimized)
- Residual CNN: Lightweight (3 layers)
- DCT: Hardware accelerated
- **Total:** Faster than HEVC

---

## 🚀 Let's Build It!

**Next immediate actions:**
1. Create residual encoder module
2. Create residual decoder module  
3. Implement hybrid codec wrapper
4. Train on GPU worker
5. Evaluate and measure 30-40 dB

**Ready to start coding!** 🎨

