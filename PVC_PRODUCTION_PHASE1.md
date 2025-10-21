# PVC v2.0 Production Path - Phase 1: Quick Wins

**Goal:** Improve from 25.06 dB → 28-30 dB PSNR  
**Timeline:** 1-2 weeks  
**Status:** Architecture Design Complete ✅

---

## 📊 Architecture Comparison

| Model | Parameters | PSNR (Target) | Status |
|-------|------------|---------------|---------|
| **Baseline SOTA** | 32.4M | 25.06 dB | ✅ Complete |
| **Production SOTA** | 92.9M | **28-30 dB** | 🔄 Training Pending |

**Improvement:** 2.9x parameters → Expected +3-5 dB improvement

---

## 🏗️ Production Architecture

### Encoder (65.9M params)
```
Input: 256x256x3 residual frame

Initial Conv: 64 channels (7x7, stride 2) → 128x128
Block 1: 64→128 channels (stride 2) → 64x64
Block 2: 128→256 channels (stride 2) → 32x32
Block 3: 256→512 channels + Attention (stride 2) → 16x16
Block 4: 512→640 channels + Attention (stride 2) → 8x8
Block 5: 640→640 channels (stride 2) → 4x4

Latent: 4x4x128 (compressed representation)
```

### Decoder (27.0M params)
```
Latent: 4x4x128

From Latent: 640 channels
Up5: 640→512 + skip + Attention → 8x8
Up4: 512→384 + skip + Attention → 16x16
Up3: 384→256 + skip → 32x32
Up2: 256→128 + skip → 64x64
Up1: 128→64 + skip → 128x128

Final Up + Conv → 256x256x3 output
```

### Key Improvements Over Baseline:
1. ✅ **Wider channels:** 640 vs 512 max
2. ✅ **More attention:** 4 attention blocks vs 2
3. ✅ **Better skip connections:** 5 levels
4. ✅ **Deeper residual blocks:** 3-layer blocks

---

## 📁 Implementation Files

**Created:**
- `/pvc_v2/models/production_residual_encoder.py` (65.9M params)
- `/pvc_v2/models/production_residual_decoder.py` (27.0M params)

**Status:** ✅ Architecture verified, forward pass tested

---

## 🎯 Next Steps

### Step 2: Generate Training Data (Pending)
- Generate 50K anime frames (vs 10K baseline)
- Use real anime clips instead of synthetic
- Data augmentation (flips, crops, color jitter)
- **Expected improvement:** +2-3 dB from better data

### Step 3: Enhanced Training (Pending)
- 200 epochs (vs 50 baseline)
- Learning rate scheduling (cosine annealing)
- Gradient accumulation for effective batch size 32
- Mixed precision training (faster, less memory)
- **Expected improvement:** +1-2 dB from better optimization

### Step 4: Launch Training (Pending)
- AWS g4dn.xlarge GPU instance
- Estimated time: 24-36 hours
- Estimated cost: $25-35
- Save checkpoints every 20 epochs

### Step 5: Evaluation (Pending)
- Calculate PSNR/SSIM on test set
- Compare with 25.06 dB baseline
- Generate comparison images
- **Target:** 28-30 dB PSNR

---

## 💰 Budget Estimate

| Item | Cost |
|------|------|
| GPU Training (g4dn.xlarge, 36 hrs) | $32 |
| Storage (models + data) | $3 |
| **Total Phase 1** | **~$35** |

---

## 📈 Success Criteria

- [ ] **Minimum:** 27 dB PSNR (+1.94 dB improvement)
- [ ] **Target:** 28-30 dB PSNR (+2.94-4.94 dB improvement)
- [ ] **Stretch:** >30 dB PSNR (+4.94 dB improvement)

If we hit **28-30 dB**, we proceed to Phase 2 (VQ-VAE hybrid) for 35-38 dB.

---

## 📝 Timeline

| Day | Task | Status |
|-----|------|--------|
| Day 1 | Architecture design | ✅ Complete |
| Day 2-3 | Data generation | ⏳ Next |
| Day 4-5 | Training | ⏳ Pending |
| Day 6 | Evaluation | ⏳ Pending |
| Day 7 | Analysis + Next Phase | ⏳ Pending |

---

**Current Status:** Architecture complete, ready for data generation and training.

**Last Updated:** Oct 21, 2025

