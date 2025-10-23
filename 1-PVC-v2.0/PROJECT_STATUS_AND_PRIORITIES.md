# PVC v2.0 - Current Status & Priorities

**Date:** October 23, 2025

---

## 🎯 Current Status

### ✅ **What We Have:**

1. **Tier 1 Hybrid Model (Trained & Working)**
   - 5.1M parameters (50 MB)
   - 48.02 dB PSNR on real anime
   - 25.2% smaller than AV1 CRF 30
   - 5.95× smaller than AV1 at matched quality (48 dB)
   - Publicly available model on S3

2. **Documentation (Complete)**
   - README with results and comparisons
   - Matched quality analysis
   - Compute requirements analysis
   - Neural CRF design (both multi-model and single-model)
   - Neural rate control design (CRF + bitrate modes)

3. **Infrastructure**
   - GitHub repo: `pvc-v2.0` branch
   - Public S3 bucket for models/results
   - Training pipeline on AWS GPU workers

---

## 🚧 **What We're Missing:**

### **Critical for Production:**

1. **❌ Single Unified Model with Variable Channels**
   - Current: Fixed 32-channel model
   - Needed: One model supporting 8-64 channels
   - Blocks: Neural CRF implementation
   - **Priority: HIGH** (enables quality control)

2. **❌ Rate Control Implementation**
   - Current: No bitrate/CRF control
   - Needed: CRF and bitrate encoding modes
   - Blocks: Streaming use cases
   - **Priority: HIGH** (professional feature)

3. **❌ Encoder/Decoder CLI Tools**
   - Current: Only test scripts
   - Needed: `pvc_encode` and `pvc_decode` commands
   - Blocks: User adoption
   - **Priority: HIGH** (usability)

### **Important for Adoption:**

4. **❌ Mobile Model (Optimized)**
   - Current: 50 MB decoder
   - Needed: 2.5-9 MB distilled/quantized model
   - Blocks: Mobile deployment
   - **Priority: MEDIUM** (enables iPhone playback)

5. **❌ Real-Time Optimizations**
   - Current: Unoptimized FP32
   - Needed: INT8 quantization, multi-threading
   - Blocks: Real-time encoding on consumer hardware
   - **Priority: MEDIUM** (enables M2/RTX 3060)

6. **❌ Container Format (.pvc file format)**
   - Current: No standard format
   - Needed: File format spec + parser
   - Blocks: Interoperability
   - **Priority: MEDIUM** (distribution)

### **Nice to Have:**

7. **❌ Video Comparison Tool**
   - Current: Manual testing
   - Needed: Automated quality comparison script
   - Blocks: Easy benchmarking
   - **Priority: LOW** (quality of life)

8. **❌ Training on More Content**
   - Current: Trained on synthetic data
   - Needed: Train on diverse real anime
   - Blocks: Generalization
   - **Priority: LOW** (current model works well)

9. **❌ Temporal Compression (Phase 3)**
   - Current: I-frames only
   - Needed: P/B frame prediction
   - Blocks: 70-90% bitrate reduction goal
   - **Priority: LOW** (future, 3-4 months)

---

## 📋 Recommended Priority Order

### **Phase A: Core Functionality (4-5 weeks)** 🔥 **START HERE**

**Goal:** Make the codec actually usable

1. **Week 1-2: Single Unified Model + Neural CRF**
   - Train variable-channel model (8, 16, 24, 32, 48, 64 channels)
   - Implement channel importance prediction
   - Test quality degradation curve
   - **Deliverable:** One 50 MB model supporting all quality levels

2. **Week 3: CLI Tools + Rate Control**
   - Implement `pvc_encode` with `--crf` and `--bitrate` flags
   - Implement `pvc_decode` 
   - Two-pass bitrate mode
   - **Deliverable:** Working command-line encoder/decoder

3. **Week 4: Container Format + Testing**
   - Design .pvc file format (header + frames + metadata)
   - Implement reader/writer
   - Test on multiple videos
   - **Deliverable:** Standard file format, can share .pvc files

4. **Week 5: Optimization (Optional)**
   - INT8 quantization of decoder weights
   - Multi-threading for tile processing
   - **Deliverable:** 2-3× speedup, mobile-ready

**Cost:** ~$100-200 (GPU training)  
**Result:** Production-ready codec users can actually use

---

### **Phase B: Mobile Deployment (2-3 weeks)** 📱

**Goal:** Enable iPhone/mobile playback

1. **Week 1: Model Distillation**
   - Train lightweight 1M param decoder (10 MB)
   - Further compress to 2.5 MB with INT8
   - Test quality loss (<2 dB)

2. **Week 2: Core ML Conversion**
   - Convert decoder to Core ML format
   - Optimize for Neural Engine
   - Test on iPhone

3. **Week 3: Mobile App (MVP)**
   - Simple iOS app for .pvc playback
   - Basic player controls
   - Publish to TestFlight

**Cost:** ~$50 (training) + time for app dev  
**Result:** iPhone 16 Pro+ can play PVC videos

---

### **Phase C: Future Enhancements (3-4 months)** 🚀

**Goal:** Advanced features for competitive edge

1. **Temporal Compression (P/B frames)**
2. **Perceptual optimization (VMAF/SSIM)**
3. **Scene detection and adaptive GOP**
4. **HDR support**
5. **4K/8K support**

**Cost:** ~$500-1000 (extensive training)  
**Result:** 70-90% bitrate reduction vs HEVC

---

## 💡 My Recommendation

### **Start with Phase A: Core Functionality**

**Why:**
1. ✅ We have a working model (Tier 1) that achieves great results
2. ✅ We have all the designs documented
3. ❌ But we can't actually USE it yet (no CLI tools, no quality control)
4. ❌ Can't share results with others (no standard format)

**What this means:**
- Don't train new models yet
- Focus on making the CURRENT model usable
- Build the infrastructure around it
- Then iterate and improve

### **Specific Next Steps (This Week):**

**Option 1: Quick Win - CLI Tools (2-3 days)**
- Create `pvc_encode.py` that uses current Tier 1 model
- Create `pvc_decode.py` 
- Define simple .pvc format (JSON header + binary data)
- **Result:** People can encode/decode videos TODAY

**Option 2: Better Foundation - Variable Channel Model (1 week)**
- Train unified model with channel importance
- Implement CRF mode (just map CRF to channel count)
- **Result:** Quality control from day 1

**Option 3: Both (2 weeks)**
- Start with CLI tools using fixed model
- Train variable model in parallel
- Swap in variable model when ready
- **Result:** Immediate usability + future flexibility

---

## 🎯 My Strong Recommendation: **Option 1 (Quick Win)**

**Start with CLI tools THIS WEEK:**

**Day 1-2:**
- [ ] Create `pvc_encode.py` using existing Tier 1 model
- [ ] Create `pvc_decode.py`
- [ ] Define .pvc format (simple: JSON header + GZIP compressed latents)

**Day 3:**
- [ ] Test encoding/decoding multiple videos
- [ ] Verify quality matches our 48 dB results
- [ ] Document usage

**Day 4-5:**
- [ ] Create comparison script (vs AV1 at multiple CRF levels)
- [ ] Generate benchmark results
- [ ] Update README with "How to Use" section

**Benefits:**
✅ Working codec in 5 days  
✅ Can share with others immediately  
✅ Start building user base  
✅ Validate architecture before investing in training  
✅ Gather feedback for what to improve next  

**Then next week:** Train variable-channel model and add CRF/bitrate control

---

## 📊 Timeline Comparison

| Approach | Week 1 | Week 2 | Week 3 | Week 4 | Usable? |
|----------|--------|--------|--------|--------|---------|
| **My Rec (Option 1)** | CLI tools | Variable model | Rate control | Polish | ✅ Day 5 |
| **Option 2** | Variable model | Variable model | CLI tools | Rate control | ✅ Week 3 |
| **Skip to Phase B** | Mobile | Mobile | Mobile | Testing | ❌ Never (no encoder) |
| **Skip to Phase C** | Temporal | Temporal | Temporal | Temporal | ❌ Never (no encoder) |

---

## ❓ Questions to Help Decide

1. **Do you want to share PVC with others soon?**
   - Yes → CLI tools first ✅
   - No → More training first

2. **Is quality control (CRF/bitrate) critical right now?**
   - Yes → Train variable model first
   - No → Fixed model is fine, add control later

3. **What's the goal for the next month?**
   - Get users/feedback → CLI tools + format
   - Perfect the codec → More training + optimization
   - Mobile launch → Phase B
   - Beat AV1 completely → Phase C

4. **Budget for GPU training?**
   - <$100 → Stick with current model, build tools
   - $100-500 → Train variable model + optimizations
   - $500+ → Full Phase A + B

---

## 🎯 TL;DR

**Current state:** Great model, no way to use it  
**Highest priority:** Build CLI tools (5 days)  
**Second priority:** Train variable-channel model (1 week)  
**Third priority:** Optimize for mobile (2-3 weeks)  

**Recommended next step:** Build encoder/decoder CLI tools this week using the existing Tier 1 model. Get something working ASAP, then iterate.

---

**What would you like to prioritize?**

A) CLI tools this week (practical, immediate results)  
B) Variable-channel model first (better foundation)  
C) Something else (tell me your goal)
