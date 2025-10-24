# Iteration 005: Hybrid Procedural-Neural Codec

**Status:** In Development  
**Date:** October 24, 2025

---

## 🎯 Goal

Create a hybrid codec that combines:
1. **Procedural generation** (anime-specific drawing functions)
2. **Neural compression** (residual autoencoder)

**Expected improvement:** 10-17 KB per frame (vs 30 KB pure neural)

---

## 🏗️ Architecture Overview

### Two-Stage Compression:

```
Input Image (512×960)
    ↓
┌─────────────────────────────────────────────┐
│ STAGE 1: Procedural Predictor (~11M params)│
│  - CNN encoder extracts image features      │
│  - Transformer generates drawing sequence   │
│  - Output: 50 operations × (func_id + 10 params)│
└─────────────────────────────────────────────┘
    ↓
Anime Renderer (differentiable)
    ↓
Procedural Base Image (512×960)
    ↓
Compute Residual = Input - Procedural Base
    ↓
┌─────────────────────────────────────────────┐
│ STAGE 2: Residual Autoencoder (~169K params)│
│  - Lightweight encoder (residual → latent)  │
│  - 32×16×30 latent (32 channels)            │
│  - Decoder reconstructs residual             │
└─────────────────────────────────────────────┘
    ↓
Final Output = Procedural Base + Residual
```

---

## 📐 Anime-Specific Functions

### **15 Anime Drawing Functions:**

#### **Line Art (0-2):**
- `anime_outline`: Thick black outlines (3px)
- `curved_outline`: Smooth curves for character silhouettes
- `tapered_line`: Hair strands, motion lines (thick→thin)

#### **Cel Shading (10-12):**
- `cel_region`: Flat color fills
- `cel_shadow`: Darker flat regions for shadows
- `cel_highlight`: Lighter flat regions for highlights

#### **Gradients (20-21):**
- `hair_gradient`: Smooth color transitions for hair
- `radial_gradient`: Eyes, cheek blush, light effects

#### **Character Features (30-32):**
- `anime_eye`: Complete eye (oval + iris + highlight)
- `anime_face`: Base face circle with outline
- `anime_mouth`: Curved mouth (smile/frown)

#### **Effects (40-42):**
- `speed_lines`: Motion lines radiating from center
- `screen_tone`: Manga dot/line patterns
- `sparkle`: 4-pointed star effects

#### **Backgrounds (50-51):**
- `sky_gradient`: Top-to-bottom sky colors
- `simple_cloud`: Overlapping circles cloud shape

---

## 📊 File Size Breakdown

| Component | Size | Format |
|-----------|------|--------|
| **Function IDs** | 50 bytes | 50 operations × 1 byte |
| **Parameters** | 1000 bytes | 50 ops × 10 params × 2 bytes (float16) |
| **Procedural total** | **1.0 KB** | Compact! |
| **Residual latent** | 60 KB | 32×16×30 float32 |
| **Residual compressed** | **15 KB** | INT8 + GZIP |
| **Total per frame** | **16 KB** | **2x better than pure neural!** |

---

## 🎓 Training Strategy

### **Phase 1: Train Procedural Predictor (2-3 days)**

**Goal:** Learn to predict anime drawing operations

**Loss function:**
```python
# Function classification loss
func_loss = CrossEntropyLoss(predicted_functions, target_functions)

# Parameter regression loss  
param_loss = MSELoss(predicted_params, target_params)

# Rendering loss (differentiable)
rendering_loss = MSELoss(rendered_procedural, target_image)

total_loss = func_loss + param_loss + 0.5 * rendering_loss
```

**Expected PSNR after Phase 1:** 12-18 dB (procedural base only)

---

### **Phase 2: Train Residual Autoencoder (1-2 days)**

**Goal:** Compress residual details

**Loss function:**
```python
# Freeze procedural predictor
with torch.no_grad():
    procedural_base = render(predict_operations(image))

# Train residual compressor
residual = image - procedural_base
residual_reconstructed = residual_autoencoder(residual)
loss = MSELoss(residual_reconstructed, residual)
```

**Expected PSNR after Phase 2:** 28-32 dB (procedural + residual)

---

### **Phase 3: Joint Fine-tuning (1 day)**

**Goal:** Optimize both together

**Loss function:**
```python
# End-to-end training
final_output = hybrid_codec(image)
loss = MSELoss(final_output, image) + perceptual_loss
```

**Expected PSNR after Phase 3:** 30-35 dB (optimized)

---

## 🚀 Quick Start

### **1. Test Model Architecture**
```bash
cd /Users/yarontorbaty/Documents/Code/AiV1/1-PVC-v2.0/training_iterations/iteration_005_hybrid_procedural
python3 hybrid_model.py
```

### **2. Test Anime Functions**
```bash
python3 test_anime_functions.py
```

### **3. Launch Training on New GPU Worker**
```bash
# (After AMI is ready and new instance launched)
cd /home/ec2-user/autoencoder_training
mkdir -p training_iterations/iteration_005_hybrid_procedural
# ... upload code ...
python3 train_hybrid_codec.py \
  --dataset /home/ec2-user/pvc_phase25/anime_frames_960x540_50k.npy \
  --batch-size 16 \
  --epochs 100 \
  --phase 1
```

---

## 📈 Expected Results vs Pure Neural

| Metric | Pure Neural (iter004) | **Hybrid (iter005)** | Improvement |
|--------|----------------------|---------------------|-------------|
| **File size** | 30 KB | **16 KB** | **47% smaller** ✅ |
| **PSNR** | 30-32 dB (expected) | **30-35 dB** | **Similar or better** ✅ |
| **Training time** | 200 epochs (~12 hrs) | **100 epochs (~6 hrs)** | **2x faster** ✅ |
| **Convergence** | Slow, plateaus | **Faster, structured** | **Better** ✅ |
| **Interpretability** | Black box | **Can visualize ops** | **Much better** ✅ |

---

## 🔬 Key Innovations

1. **Anime-specific functions** instead of generic primitives
2. **Two-stage compression** (structure + details)
3. **Differentiable rendering** enables end-to-end training
4. **Residual compression** is easier than full image
5. **Structured approach** should converge faster

---

## 📝 Files

- `anime_functions.py`: 15 anime-specific drawing functions
- `hybrid_model.py`: Complete hybrid codec architecture
- `train_hybrid_codec.py`: Training script (to be created)
- `README.md`: This file

---

## 🎯 Next Steps

1. ✅ Design anime functions (done!)
2. ✅ Create hybrid model architecture (done!)
3. ⏳ Wait for AMI to complete
4. ⏳ Launch new GPU instance
5. ⏳ Create training script
6. ⏳ Start Phase 1 training (procedural predictor)
7. ⏳ Evaluate and proceed to Phase 2

---

**Last Updated:** October 24, 2025  
**Status:** Architecture complete, ready for training once AMI/instance ready

