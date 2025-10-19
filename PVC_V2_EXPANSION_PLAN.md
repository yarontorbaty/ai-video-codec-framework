# PVC v2.0 Expansion Plan

**Goal:** Achieve PSNR 25-35 dB and SSIM > 0.85 for anime/animation content  
**Current Status:** PSNR 4.06 dB, SSIM 0.17  
**Required Improvement:** ~7-9x PSNR increase

---

## 🎯 Strategy: Three-Phase Expansion

### Phase 1: Expand Function Library (50+ functions)
**Target:** PSNR 10-15 dB  
**Timeline:** 2-3 hours  
**Effort:** Medium

#### New Function Categories:

1. **Advanced Fill Functions (10 new)**
   - `fill_radial_gradient` - radial color transitions
   - `fill_conic_gradient` - angular color transitions
   - `fill_noise_perlin` - Perlin noise texture
   - `fill_noise_simplex` - Simplex noise texture
   - `fill_checkerboard` - checkerboard pattern
   - `fill_dots` - dot pattern
   - `fill_stripes` - stripe pattern
   - `fill_wave` - wave pattern
   - `fill_cellular` - cellular/Voronoi pattern
   - `fill_texture_blend` - blend two textures

2. **Advanced Shape Functions (15 new)**
   - `draw_polygon` - arbitrary polygon
   - `draw_bezier_curve` - cubic Bezier curves
   - `draw_arc` - circular arc
   - `draw_rounded_rect` - rounded rectangle
   - `draw_star` - n-pointed star
   - `draw_heart` - heart shape
   - `draw_triangle` - triangle
   - `draw_trapezoid` - trapezoid
   - `draw_parallelogram` - parallelogram
   - `draw_ring` - ring/donut shape
   - `draw_crescent` - crescent moon
   - `draw_cross` - cross shape
   - `draw_arrow` - arrow
   - `draw_speech_bubble` - speech bubble
   - `draw_cloud` - cloud shape

3. **Texture & Effect Functions (15 new)**
   - `apply_blur` - Gaussian blur region
   - `apply_sharpen` - sharpen region
   - `apply_glow` - glow effect
   - `apply_shadow` - drop shadow
   - `apply_emboss` - emboss effect
   - `apply_posterize` - posterization
   - `apply_pixelate` - pixelation effect
   - `apply_oil_paint` - oil paint filter
   - `apply_watercolor` - watercolor effect
   - `apply_outline` - outline extraction
   - `apply_halftone` - halftone pattern
   - `apply_dither` - dithering
   - `apply_scanlines` - scanline effect
   - `apply_chromatic_aberration` - chromatic aberration
   - `apply_vignette` - vignette effect

4. **Compositing Functions (10 new)**
   - `blend_normal` - normal blending
   - `blend_multiply` - multiply blending
   - `blend_screen` - screen blending
   - `blend_overlay` - overlay blending
   - `blend_add` - additive blending
   - `blend_subtract` - subtractive blending
   - `mask_alpha` - alpha masking
   - `mask_luminance` - luminance masking
   - `composite_over` - alpha compositing
   - `composite_xor` - XOR compositing

**Total New Functions:** 50  
**Total Functions:** 60 (10 existing + 50 new)

---

### Phase 2: Increase Sequence Length & Architecture
**Target:** PSNR 18-25 dB  
**Timeline:** 2-3 hours  
**Effort:** High

#### Architecture Changes:

1. **Longer Sequences:**
   - Increase max sequence length: 5 → 50
   - Allow model to use 10x more function calls per frame
   - Better granularity for reconstruction

2. **Attention Mechanism:**
   - Add multi-head attention to RNN decoder
   - Better long-range dependencies
   - Selective focus on important regions

3. **Hierarchical Structure:**
   - Level 1: Coarse (5-10 functions) - background, large regions
   - Level 2: Medium (15-25 functions) - objects, characters
   - Level 3: Fine (30-50 functions) - details, edges, highlights
   - Train in stages, progressively refining

4. **Per-Function Parameter Predictors:**
   - Current: Single 10-param output for all functions
   - New: Custom parameter predictor per function type
   - Example: `draw_polygon` needs variable number of points
   - Example: `apply_blur` needs radius, sigma

5. **Larger Model:**
   - CNN: ResNet-34 → ResNet-50 or EfficientNet-B3
   - RNN: GRU → Transformer decoder
   - Hidden dim: 128 → 256
   - More capacity for complex scenes

---

### Phase 3: Advanced Training & Data
**Target:** PSNR 25-35 dB, SSIM > 0.85  
**Timeline:** 3-4 hours  
**Effort:** High

#### Training Improvements:

1. **Real Anime Data:**
   - Use actual anime frames from provided clips
   - Extract ground truth function sequences via inverse rendering
   - Train on 5K-10K real anime frames
   - Better generalization to target content

2. **Perceptual Loss:**
   - Add VGG-based perceptual loss
   - Focus on visual similarity, not pixel-wise MSE
   - Weight: 0.5 × function loss + 0.3 × param loss + 0.2 × perceptual loss

3. **Progressive Training:**
   - Stage 1: 10 functions, 5 seq length, 1K samples (baseline)
   - Stage 2: 30 functions, 20 seq length, 3K samples
   - Stage 3: 60 functions, 50 seq length, 10K samples
   - Each stage fine-tunes previous stage

4. **Data Augmentation:**
   - Color jittering
   - Random crops, scales, rotations
   - Brightness/contrast adjustment
   - Better robustness

5. **Multi-GPU Training:**
   - Use multiple GPU workers in parallel
   - Distributed data parallel (DDP)
   - Faster iteration, larger batch sizes

---

## 📦 Implementation Roadmap

### Week 1: Foundation (You Are Here)
- [x] Phase 0: Proof of concept with parameter supervision
- [x] Results: PSNR 4.06 dB baseline established

### Week 2: Expansion
- [ ] **Day 1-2:** Implement 50 new graphics functions
- [ ] **Day 3-4:** Extend architecture (attention, longer sequences)
- [ ] **Day 5-6:** Train and evaluate on synthetic data
- [ ] **Day 7:** Real anime data pipeline

### Week 3: Optimization
- [ ] **Day 1-2:** Perceptual loss and advanced training
- [ ] **Day 3-4:** Progressive training stages
- [ ] **Day 5-6:** Multi-GPU distributed training
- [ ] **Day 7:** Final evaluation and tuning

### Week 4: Validation
- [ ] **Day 1-2:** Test on all anime clips
- [ ] **Day 3-4:** Quality analysis (PSNR, SSIM, VMAF)
- [ ] **Day 5-6:** Compression benchmarks vs AV1
- [ ] **Day 7:** Documentation and demos

---

## 🎯 Success Metrics

| Metric | Current | Target | Stretch Goal |
|--------|---------|--------|--------------|
| PSNR | 4.06 dB | 25 dB | 35 dB |
| SSIM | 0.17 | 0.85 | 0.92 |
| Compression vs AV1 | 99.8% | 90% | 95% |
| Functions per frame | 3-5 | 30-50 | 100+ |
| Function library size | 10 | 60 | 100+ |
| Training time | 2.7 min | <30 min | <60 min |
| Inference time/frame | N/A | <100ms | <50ms |

---

## 💰 Resource Requirements

### Compute:
- **GPU Worker:** g4dn.xlarge (current) → g4dn.2xlarge or g5.2xlarge
  - More GPU memory for larger models
  - Faster training (T4 → A10G)
- **Storage:** 50GB → 100GB EBS (more training data)
- **Training Time:** ~10-15 GPU hours total

### Development:
- **Code:** ~2000 lines of new Python code
  - 50 new function implementations
  - Architecture modifications
  - Training pipeline enhancements
- **Testing:** Each function needs unit tests
- **Documentation:** Function library reference

---

## 🚀 Quick Start: Phase 1

Let's begin with **expanding the function library** to 60 functions. This is the highest ROI task.

### Immediate Next Steps:

1. **Create new functions module:** `pvc_v2/graphics/primitives_extended.py`
2. **Implement 10 functions at a time** (test each batch)
3. **Update synthetic generator** to use new functions
4. **Retrain model** with expanded function set
5. **Measure improvement** (expect 2-3x PSNR increase)

### Function Implementation Priority:

**High Priority (implement first):**
1. Advanced shapes (polygon, bezier, arc, rounded rect)
2. Advanced gradients (radial, conic)
3. Basic textures (noise, patterns)
4. Blur and shadow effects

**Medium Priority:**
5. Blending modes
6. Texture effects
7. Specialized shapes

**Low Priority:**
8. Advanced effects (oil paint, watercolor)
9. Complex compositing

---

## 🤔 Alternative: Pivot to Hybrid?

If after Phase 1 we don't see 3-4x PSNR improvement (12-16 dB), we should consider:

**Hybrid Approach:**
- Use PVC for coarse structure (5-10 functions)
- Add neural residual encoder for fine details
- Target: 85% compression from PVC + 10% from residuals = 95% total
- Expected PSNR: 30-40 dB (residuals fill the gap)

This gives us a **fallback strategy** if pure procedural doesn't reach quality targets.

---

## ✅ Decision Point

**Proceed with Phase 1?**
- Implement 50 new graphics functions
- Test on synthetic data
- Measure PSNR improvement
- Reassess after results

**Estimated time:** 3-4 hours for implementation + testing  
**Expected result:** PSNR 10-15 dB (2-3x improvement)

If approved, I'll start implementing the extended function library now.

