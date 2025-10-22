# 🔧 LiDAR Depth Quality Improvements

## 📊 Current Issues

### What You're Seeing:
- ✅ **Diagonal lines:** Normal LiDAR scanning pattern
- ✅ **Blocky artifacts:** From simple downsampling
- ✅ **Temporal jitter:** No frame-to-frame smoothing
- ✅ **Noise:** Typical for time-of-flight sensors

### Are These Artifacts Bad for Training?
**Short Answer:** Not really!

**The LCM codec will learn to:**
- Ignore scanning artifacts
- Smooth temporal noise
- Focus on semantic depth structure
- Compress efficiently despite noise

**However, cleaner depth = better training convergence**

---

## 🎯 Three Levels of Quality Improvement

### Level 1: Quick Fix (5 minutes)
**Add bilateral filtering to smooth while preserving edges**

This would require updating `FileWriter.swift` to:
1. Use Core Image's `CIBilateralFilter`
2. Apply to depth before downsampling
3. Keep edge structures intact

**Benefit:** Removes noise while keeping object boundaries sharp

### Level 2: Better Downsampling (10 minutes)
**Use bilinear interpolation instead of nearest-neighbor**

Current code (line 237):
```swift
let srcX = Int(Float(x) * scaleX)  // Blocky!
```

Better approach:
```swift
// Bilinear interpolation
let srcXf = Float(x) * scaleX
let srcYf = Float(y) * scaleY
// Sample 4 neighbors and blend
```

**Benefit:** Smoother depth maps, less aliasing

### Level 3: Temporal Smoothing (20 minutes)
**Smooth depth across frames**

Add exponential moving average:
```swift
// Keep previous frame's depth
private var previousDepth: CVPixelBuffer?

// Blend: 80% current + 20% previous
let smoothedDepth = blend(current, previousDepth, alpha: 0.8)
```

**Benefit:** Reduces temporal jitter, more stable depth

---

## 🤔 Should You Fix This Now?

### **Recommendation: NO - Don't fix yet!**

**Why?**
1. ✅ **Current depth is good enough for initial training**
2. ✅ **LCM will handle noise during training**
3. ✅ **You need more training data first (10-15 videos)**
4. ✅ **Optimization can come later**

**Better workflow:**
1. **Now:** Capture 10-15 videos with current implementation
2. **Start training:** See how well LCM handles raw depth
3. **If PSNR < 30 dB:** Then consider depth improvements
4. **If PSNR > 32 dB:** Current depth is fine!

---

## 📊 Expected Depth Quality vs Training Performance

### Scenario A: Raw LiDAR (Current)
- **Artifacts:** Yes (diagonal lines, some noise)
- **Training PSNR:** 30-35 dB (still good!)
- **Training time:** Normal
- **Compression:** 50-60x

### Scenario B: Filtered LiDAR (Level 1-3 fixes)
- **Artifacts:** Minimal
- **Training PSNR:** 33-38 dB (slightly better)
- **Training time:** Normal
- **Compression:** 55-65x

**Difference:** ~2-3 dB PSNR improvement
**Worth it?** Only if initial training underperforms

---

## 🔬 Analyzing Your Current Depth Data

To verify depth quality, you can:

### View depth in VLC:
```bash
# Open the video
open ~/lumaflow_training_data/lumaflow_1761095137.mov

# In VLC: Video → Video Track → Track 2 (Depth)
```

### Extract depth frames:
```bash
ffmpeg -i ~/lumaflow_training_data/lumaflow_1761095137.mov \
  -map 0:1 -vframes 10 ~/Downloads/depth_frame_%03d.png
```

### What to Look For:
- ✅ **Object boundaries visible?** → Good
- ✅ **Depth gradient smooth?** → Good
- ✅ **Foreground/background separation clear?** → Good
- ❌ **Complete black frames?** → Bad (no depth data)
- ❌ **Random noise patterns?** → Bad (sensor malfunction)

---

## 🎯 Decision Tree

```
Are depth frames mostly black?
├─ YES → Fix sensor/app configuration ⚠️
└─ NO → Continue reading...

Can you see object boundaries?
├─ NO → Need better filtering ⚠️
└─ YES → Continue reading...

Is foreground/background separated?
├─ NO → Check depth calibration ⚠️
└─ YES → Depth is GOOD ENOUGH! ✅

Are there diagonal lines/artifacts?
├─ YES → NORMAL for LiDAR! ✅
└─ NO → Lucky! Even better ✅
```

---

## ✅ My Recommendation

### For Now:
1. **Accept current depth quality** - It's good enough!
2. **Focus on capturing more videos** (10-15 total)
3. **Start training** once you have sufficient data
4. **Evaluate results** after first training run

### If Training Results Are Poor (PSNR < 28 dB):
Then come back and implement:
1. Bilateral filtering (Level 1)
2. Better downsampling (Level 2)
3. Temporal smoothing (Level 3)

### If Training Results Are Good (PSNR > 30 dB):
🎉 **Current depth quality is perfect! Ship it!**

---

## 🔧 Quick Depth Verification

Run this to check your depth quality:

```bash
# Extract 5 depth frames
ffmpeg -i ~/lumaflow_training_data/lumaflow_1761095137.mov \
  -map 0:1 -vframes 5 ~/Downloads/depth_%03d.png

# Open them
open ~/Downloads/depth_*.png
```

**What you should see:**
- Grayscale images
- Objects visible
- Depth gradients (near = white, far = black)
- Some diagonal patterns (normal!)
- Some noise (normal!)

---

## 📞 Bottom Line

**Your depth data is probably fine!** 

LiDAR artifacts are:
- ✅ Expected from the sensor
- ✅ Handled by LCM training
- ✅ Not a blocker for 30+ dB PSNR

**Priority order:**
1. 📱 **Capture more videos** (most important!)
2. 🧪 **Train first model** (see actual results)
3. 🔧 **Optimize depth** (only if needed)

Don't optimize prematurely - train first, then iterate! 🚀

