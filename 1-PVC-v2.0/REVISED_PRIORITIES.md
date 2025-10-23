# Revised Priorities: AV1 Integration First

**Date:** October 23, 2025

---

## 🎯 Key Insight

**You're absolutely right:** I-frame-only codecs aren't useful for anime/animation in practice.

### Why:
- **Anime has high temporal redundancy** (static backgrounds, limited motion)
- **P/B frames are 50-100× more efficient** than I-frames for anime
- **Real-world usage:** 90-95% of frames are P/B frames, only 5-10% are I-frames
- **Net benefit of I-frame only:** Minimal (~2-5% overall savings)

### What People Actually Need:
A **complete codec** that handles:
- ✅ I-frames (keyframes) - our neural codec
- ✅ P/B frames (motion compensation) - AV1's temporal prediction
- ✅ Integrated workflow - one command to encode

---

## 📋 New Priority: AV1 Hybrid Integration

### Goal:
Replace AV1's I-frames with our neural codec, keep AV1's excellent temporal compression.

### Expected Result:
- **I-frame compression:** 25% smaller (50 KB vs 67 KB for 1080p)
- **Overall bitrate:** ~2-5% improvement (since I-frames are only 5-10% of data)
- **P/B frames:** Unchanged (AV1 is already excellent at this)

---

## 🛠️ Implementation Options

### **Option 1: Quick Proof-of-Concept (RECOMMENDED)** ⭐
**Time:** 4-6 hours  
**Cost:** $0

**Approach:** External preprocessing pipeline

```bash
# Step 1: Extract keyframes from source video
ffmpeg -i input.mp4 -vf "select='eq(pict_type,I)'" -vsync 0 iframes_%04d.png

# Step 2: Encode I-frames with our neural codec
python neural_encode_iframes.py --input iframes/ --output neural/

# Step 3: Encode P/B frames with AV1
ffmpeg -i input.mp4 -c:v libaom-av1 -b:v 10M -skip_frame nokey output_temp.mp4

# Step 4: Mux neural I-frames + AV1 P/B frames
python mux_hybrid.py --neural neural/ --av1 output_temp.mp4 --output hybrid.mp4
```

**Pros:**
✅ Working today (no new training)  
✅ Prove the concept quickly  
✅ Measure real-world benefits  
✅ No AV1 source code changes  

**Cons:**
❌ Manual multi-step workflow  
❌ Not production-ready  
❌ Requires custom tooling  

---

### **Option 2: FFmpeg Filter Plugin**
**Time:** 2-3 days  
**Cost:** $0

Create a custom FFmpeg filter that calls our neural codec for I-frames.

**Approach:**
1. Create `vf_neural_iframe` FFmpeg filter
2. Wrap PyTorch model with LibTorch C++ API
3. Compile custom FFmpeg build

**Usage:**
```bash
ffmpeg -i input.mp4 -vf neural_iframe -c:v libaom-av1 -b:v 10M output.mp4
# Automatically uses neural codec for I-frames, AV1 for P/B
```

**Pros:**
✅ One-command encoding  
✅ Integrates with FFmpeg ecosystem  
✅ Standard workflow  

**Cons:**
❌ Requires C++ coding  
❌ FFmpeg compilation needed  
❌ LibTorch C++ wrapper (complex)  

---

### **Option 3: Modify libaom (Production-Ready)**
**Time:** 1-2 weeks  
**Cost:** $0

Directly integrate into AV1 encoder.

**Approach:**
1. Fork libaom
2. Replace I-frame encoder with neural codec
3. Keep AV1 rate control and temporal prediction

**Pros:**
✅ True integration  
✅ Production-ready  
✅ Can contribute back to libaom  

**Cons:**
❌ Requires deep AV1 knowledge  
❌ 1-2 weeks development  
❌ Ongoing maintenance  

---

## 🎯 Recommended Path Forward

### **This Week: Proof of Concept (Option 1)**

**Monday-Tuesday (4-6 hours):**
1. Create `neural_iframe_encoder.py` - encodes I-frames with Tier 1 model
2. Create `extract_iframes.py` - extracts keyframes from video
3. Create `hybrid_muxer.py` - combines neural I-frames + AV1 P/B frames
4. Test on 2-3 anime clips

**Wednesday (2 hours):**
5. Measure actual bitrate improvement
6. Compare quality (PSNR, SSIM, VMAF)
7. Document results

**Thursday-Friday (Optional):**
8. If results are good → Start FFmpeg filter (Option 2)
9. If results are underwhelming → Re-evaluate approach

---

### **Next 2 Weeks: Production Integration (Option 2 or 3)**

**If proof-of-concept shows >2% overall improvement:**
- Proceed with FFmpeg filter or libaom integration
- Create proper CLI tools
- Document workflow

**If proof-of-concept shows <2% improvement:**
- Re-think strategy
- Maybe neural codec alone is better for I-frame-heavy use cases
- Or focus on temporal neural codec instead

---

## 📊 Expected Results

### Realistic Expectations:

**For typical anime encode (30 fps, 1 minute):**
- Total frames: 1,800
- I-frames (every 60 frames): 30 frames (~2%)
- P/B frames: 1,770 frames (~98%)

**Bitrate breakdown:**
```
I-frames: 30 × 67 KB = 2,010 KB (AV1)
          30 × 50 KB = 1,500 KB (Neural) → 25% reduction
          Savings: 510 KB

P/B frames: 1,770 × 1.5 KB = 2,655 KB (unchanged)

Total video:
  AV1 only: 4,665 KB
  Hybrid:   4,155 KB
  
Net improvement: 11% overall bitrate reduction
```

**For I-frame heavy content (scene changes every 2 seconds):**
```
I-frames: 30 frames/min → 25% of data
Savings: ~6-8% overall
```

---

## 💡 Key Decision Points

### After Proof-of-Concept:

**If >5% improvement:**
→ **Worth it!** Proceed with production integration

**If 2-5% improvement:**
→ **Marginal.** Consider if complexity is worth it

**If <2% improvement:**
→ **Not worth it.** Focus on:
- Pure neural codec (I-frame only) for broadcast use cases
- OR temporal neural codec (replace P/B frames too)
- OR niche use cases (I-frame heavy content)

---

## 🚀 Immediate Action Plan

**Today/Tomorrow:**

1. **Create neural I-frame encoder** (2 hours)
   - Load Tier 1 model
   - Encode 960×540 tiles
   - Save as compressed format

2. **Create extraction/muxing tools** (2 hours)
   - Extract I-frames from video
   - Mux neural + AV1 streams

3. **Test on real anime** (1 hour)
   - Encode 3 test clips
   - Measure bitrate and quality

4. **Measure & decide** (1 hour)
   - Calculate actual savings
   - Decide: Continue or pivot?

**Total time: 6 hours to know if this approach is viable**

---

## 🔄 Alternative: Temporal Neural Codec

**If AV1 integration doesn't show strong results, consider:**

### Hybrid Temporal Codec:
- Neural I-frames (our current model)
- Neural P-frames (train motion prediction model)
- Skip AV1 entirely

**Benefits:**
- Full control over codec
- Can optimize specifically for anime
- Could achieve 50-70% reduction vs AV1 (not just 11%)

**Challenges:**
- Need to train temporal model (3-4 weeks)
- More complex than AV1 integration
- Higher risk

**Timeline:** 4-6 weeks for initial results

---

## 📝 Summary

**Your insight is spot-on:** I-frame-only isn't practical for anime.

**Recommended immediate action:**
1. ✅ Build AV1 hybrid proof-of-concept (6 hours)
2. ✅ Measure real-world improvement
3. ✅ Decide: Continue integration OR pivot to temporal neural codec

**This week's goal:**
Validate whether AV1 I-frame replacement provides meaningful real-world benefits (>5% overall).

**Next week's goal:**
If validated → Production integration  
If not → Design temporal neural codec (P/B frames)

---

**Should we start with the proof-of-concept?**

A) Yes, build the AV1 hybrid PoC today/tomorrow (6 hours)  
B) Skip to temporal neural codec (full control, 4-6 weeks)  
C) Something else?
