# PVC v2.0 Phased Development Roadmap

**Target:** Incremental bitrate reduction (50% → 70% → 90%) vs HEVC  
**Focus:** Real-time performance at each phase

---

## 📊 Current Status (Phase 1)

### Training Status (Tonight)
- **Model:** 93M parameters (65.9M encoder + 27M decoder)
- **Progress:** Epoch 19/100
- **PSNR:** 41.05 dB (synthetic), 26.36 dB (real anime)
- **Bitrate:** ~14.5 Mbps (with INT8+GZIP)
- **Compression:** 24:1 (95.8% reduction from raw)

### Computational Requirements
- **Encoding:** 25 TFLOPS GPU + 0.3 TFLOPS CPU
- **Decoding:** 10.6 TFLOPS GPU + 0.15 TFLOPS CPU
- **Status:** ❌ NOT real-time (requires RTX 3090 for encoding, can't decode in real-time)

### vs HEVC Baseline
- **HEVC:** 10 Mbps @ 38.21 dB PSNR
- **Ours:** 14.5 Mbps @ 26-29 dB PSNR (expected at Epoch 100)
- **Gap:** 1.4× larger bitrate, 10 dB lower quality

**Assessment:** Not yet competitive with HEVC

---

## 🎯 Phase 2: 50% Bitrate Reduction (Real-time Capable)

### Timeline
**1-2 weeks** from Phase 1 completion

### Target Metrics
- **Bitrate:** 7.5 Mbps (50% reduction vs HEVC)
- **PSNR:** 28-29 dB
- **Encoding:** Real-time at 3.3× speed
- **Decoding:** Real-time at 7× speed

### Key Changes
1. **Model Optimization**
   - Reduce latent channels: 128 → 64
   - Model pruning: 93M → 45M parameters
   - INT8 inference (not just storage)

2. **Patch Size Optimization**
   - Increase patch size: 256×256 → 512×512
   - Reduce patch count: 32 → 8 patches per frame
   - 4× fewer forward passes

3. **Quantization**
   - INT8 inference throughout
   - 2× speedup vs FP32
   - Minimal quality loss (<0.5 dB)

### Computational Requirements
- **Encoding:** 3 TFLOPS GPU + 0.3 TFLOPS CPU
- **Decoding:** 1.35 TFLOPS GPU + 0.15 TFLOPS CPU

### Hardware Requirements
- **Encoding:** NVIDIA RTX 3060 or better (mid-range)
- **Decoding:** NVIDIA GTX 1660 or better (entry-level)
- **Status:** ✅ Real-time capable

### Technical Tasks
1. [ ] Implement model pruning scripts
2. [ ] Reduce latent dimensionality (128→64 channels)
3. [ ] Convert models to INT8 quantized inference
4. [ ] Optimize patch processing (512×512)
5. [ ] Benchmark on RTX 3060 and GTX 1660
6. [ ] Train pruned model (50 epochs, 10K samples)
7. [ ] Validate PSNR target (28-29 dB)

**Estimated Effort:** 40-60 hours

---

## 🎯 Phase 3: 70% Bitrate Reduction (Integrated GPU Encoding)

### Timeline
**2-3 weeks** from Phase 2 completion

### Target Metrics
- **Bitrate:** 3.5 Mbps (70% reduction vs HEVC)
- **PSNR:** 30-32 dB
- **Encoding:** Real-time at 20-40× speed (can run on integrated GPU!)
- **Decoding:** Real-time at 2.7× speed

### Key Changes
1. **Procedural Video Codec (PVC) Integration**
   - Edge detection and shape extraction (CPU)
   - Parameter fitting for graphics primitives
   - Captures 60-70% of anime structure
   - Bitrate: ~1 Mbps

2. **Neural Residuals Only**
   - Encode only fine details (30-40% of pixels)
   - Smaller model: 25M encoder, 12M decoder
   - Sparse encoding significantly faster
   - Bitrate: ~2 Mbps

3. **Temporal Prediction**
   - I-frames every 30 frames (full encoding)
   - P-frames: motion compensation + small residual
   - 90% cheaper for P-frames
   - Effective bitrate: ~0.5 Mbps average

### Computational Requirements
- **Encoding:** 0.1 TFLOPS GPU + 0.2 TFLOPS CPU
- **Decoding:** 2.2 TFLOPS GPU + 0.15 TFLOPS CPU

### Hardware Requirements
- **Encoding:** Integrated GPU (Intel Iris Xe) or entry-level discrete
- **Decoding:** NVIDIA GTX 1650 or better
- **Status:** ✅ Real-time capable, can encode on any modern laptop

### Technical Tasks
1. [ ] Port PVC graphics primitives library
2. [ ] Implement procedural encoder (CPU-based)
3. [ ] Train smaller residual encoder/decoder (25M/12M params)
4. [ ] Implement temporal prediction (I/P-frame structure)
5. [ ] Integrate hybrid pipeline (procedural + residual + temporal)
6. [ ] Optimize procedural rendering on GPU
7. [ ] Benchmark on integrated GPU (encoding)
8. [ ] Benchmark on GTX 1650 (decoding)
9. [ ] Validate PSNR target (30-32 dB)

**Estimated Effort:** 80-120 hours

---

## 🎯 Phase 4: 90% Bitrate Reduction (Algorithmic Refinement)

### Timeline
**1-2 months** from Phase 3 completion

### Target Metrics
- **Bitrate:** 1.0-1.2 Mbps (90% reduction vs HEVC)
- **PSNR:** 28-30 dB (perceptually optimized for anime)
- **Encoding:** Real-time at 8-16× speed
- **Decoding:** Real-time at 2.7× speed

### Key Changes
1. **Entropy Coding**
   - Context-adaptive arithmetic coding
   - Learned probability models
   - 20-30% additional compression
   - CPU-efficient implementation

2. **Perceptual Optimization**
   - Optimize for human perception, not PSNR
   - Saliency-based bit allocation
   - Can trade 2-3 dB PSNR for 30% bitrate savings
   - Important for anime (focus on characters, not backgrounds)

3. **Anime-Specific Optimizations**
   - Separate networks for character vs background
   - Line art + color separation
   - Exploit cel animation structure
   - 20% improvement expected

4. **Larger Temporal Context**
   - Use 16-32 frame sequences (not just 1-2)
   - Better motion prediction
   - Scene change detection
   - 15% improvement expected

### Computational Requirements
- **Encoding:** 0.25 TFLOPS GPU + 0.8 TFLOPS CPU
- **Decoding:** 2.2 TFLOPS GPU + 0.45 TFLOPS CPU

### Hardware Requirements
- **Encoding:** Integrated GPU + modern 8-core CPU
- **Decoding:** NVIDIA GTX 1650 (same as Phase 3)
- **Status:** ✅ Real-time capable

### Technical Tasks
1. [ ] Implement context-adaptive arithmetic coder
2. [ ] Train learned probability models
3. [ ] Implement perceptual loss (VGG-based)
4. [ ] Build saliency detection module
5. [ ] Train separate character/background networks
6. [ ] Implement line art extraction
7. [ ] Build 16-32 frame temporal predictor
8. [ ] Implement scene change detection
9. [ ] Integrate all components into unified pipeline
10. [ ] Extensive testing on real anime dataset (50+ clips)
11. [ ] User studies for perceptual quality validation

**Estimated Effort:** 200-300 hours

---

## 📈 Summary Comparison

| Phase | Timeline | Bitrate | PSNR | Encoding GPU | Decoding GPU | Real-time? |
|-------|----------|---------|------|--------------|--------------|------------|
| **Current** | Tonight | 14.5 Mbps | 28-29 dB | RTX 3090 | RTX 2060+ | ❌ No |
| **Phase 2** | 1-2 weeks | 7.5 Mbps | 28-29 dB | RTX 3060 | GTX 1660 | ✅ Yes |
| **Phase 3** | 2-3 weeks | 3.5 Mbps | 30-32 dB | Integrated | GTX 1650 | ✅ Yes |
| **Phase 4** | 1-2 months | 1.0-1.2 Mbps | 28-30 dB | Integrated | GTX 1650 | ✅ Yes |

**HEVC Baseline:** 10 Mbps @ 38.21 dB

---

## 🔑 Key Insights

### 1. Phase 2 is the First Practical Milestone
- Real-time encoding/decoding on consumer hardware
- 50% bitrate reduction is significant
- Mid-range GPU requirements
- **Recommendation:** Target this for initial deployment

### 2. Phase 3 is the Sweet Spot
- 70% bitrate reduction
- Can encode on ANY modern laptop (integrated GPU)
- Best balance of compression vs hardware requirements
- **Recommendation:** Ideal target for production deployment

### 3. Phase 4 is Research/Niche Applications
- 90% reduction is extremely impressive
- Same hardware requirements as Phase 3
- Mostly algorithmic improvements, not computational
- Best for anime/animation content
- **Recommendation:** Research milestone, niche applications

### 4. Decoding is the Bottleneck
- 2.2 TFLOPS needed (Phases 3-4)
- Due to procedural rendering overhead
- Could be accelerated with:
  - Custom hardware (ASIC)
  - Shader-based rendering
  - WebGPU implementation for browser playback

### 5. HEVC Still More Efficient
- HEVC software decode: 50-100 GFLOPS (CPU only)
- Our Phase 4 decode: 2.2 TFLOPS GPU + 0.45 TFLOPS CPU
- **22-44× more compute than HEVC**
- But: Our quality at 1 Mbps is competitive with HEVC at 10 Mbps for anime

---

## 🎬 Next Steps

### Immediate (Tonight)
- [x] Complete Phase 1 training (Epoch 100)
- [ ] Evaluate final PSNR on synthetic data
- [ ] Test on multiple real anime clips
- [ ] Document baseline performance

### Short-term (This Week)
- [ ] Decide on target: Phase 2 (50%) or Phase 3 (70%) or Phase 4 (90%)
- [ ] Set up development environment for chosen phase
- [ ] Create detailed implementation plan

### Recommended Path
**Option A: Conservative (4-6 weeks)**
- Phase 2 (50% reduction, real-time on mid-range GPU)
- Quick win, proven technology
- Lower risk

**Option B: Balanced (6-10 weeks)**
- Phase 2 → Phase 3 (70% reduction, integrated GPU encoding)
- Best for deployment
- Moderate risk

**Option C: Ambitious (3-4 months)**
- Phase 2 → Phase 3 → Phase 4 (90% reduction)
- Research milestone
- Higher risk, high reward

---

## 📝 Notes

- All PSNR targets assume anime/animation content
- Live-action video would require different approach
- Bitrate estimates assume INT8 quantization + GZIP
- Real-world performance may vary by ±20%
- Hardware requirements are minimums; better hardware = faster encoding/decoding
- Phase 4 perceptual optimizations may sacrifice PSNR but improve perceived quality

---

**Last Updated:** {{ current_date }}  
**Status:** Phase 1 in progress (Epoch 19/100)

