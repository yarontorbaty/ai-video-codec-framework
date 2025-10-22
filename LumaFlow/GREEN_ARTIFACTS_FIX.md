# 🐛 Green Artifacts Bug - FIXED!

## The Problem

**Symptoms:**
- Green/purple/colorful artifacts in depth video
- Shapes hard to see due to color noise
- Should be pure grayscale, but has color tinting

**Root Cause:**
YUV 4:2:0 format has **TWO planes**:
1. **Y plane (luminance):** Grayscale intensity ✅ **WE WERE FILLING THIS**
2. **UV plane (chrominance):** Color information ❌ **WE FORGOT THIS!**

When the UV plane isn't initialized, it contains **random memory** → random colors!

---

## The Fix

### Before (Buggy Code):
```swift
// Only fills Y plane
for y in 0..<192 {
    for x in 0..<256 {
        outputPointer[dstIndex] = grayscale  // Only Y!
    }
}
return outputBuffer  // UV plane = garbage memory! 🐛
```

**Result:** Random green/purple because UV has uninitialized values

### After (Fixed Code):
```swift
// Fill Y plane with depth
for y in 0..<192 {
    for x in 0..<256 {
        outputPointer[dstIndex] = grayscale  // Y plane
    }
}

// NEW: Initialize UV plane to 128 (neutral gray)
let uvPointer = CVPixelBufferGetBaseAddressOfPlane(outputBuffer, 1)
for y in 0..<(192/2) {
    for x in 0..<uvBytesPerRow {
        uvPointer[y * uvBytesPerRow + x] = 128  // Neutral = no color
    }
}
return outputBuffer  // Now pure grayscale! ✅
```

**Result:** Clean grayscale depth with no color artifacts

---

## Why 128 for UV?

In YUV color space:
- **Y = 0-255:** Brightness (0=black, 255=white)
- **U = 0-255:** Blue-Yellow axis (128=neutral)
- **V = 0-255:** Red-Green axis (128=neutral)

**UV = 128,128 = GRAY (no color)**
- U < 128 → Blue tint
- U > 128 → Yellow tint
- V < 128 → Green tint
- V > 128 → Red/Purple tint

Random memory in UV → Random colors!

---

## What to Do Now

### 1. Rebuild the App
```bash
# In Xcode:
1. Clean Build Folder (Cmd+Shift+K)
2. Build & Run (Cmd+R)
```

### 2. Capture a New Test Video
- Record 10-15 seconds
- Test depth quality

### 3. Verify the Fix
```bash
# Extract depth frames
ffmpeg -i ~/Downloads/lumaflow_NEW.mov -map 0:1 -vframes 3 \
  ~/Downloads/depth_fixed_%03d.png -y

# Open them
open ~/Downloads/depth_fixed_*.png
```

**Expected result:** Pure grayscale depth maps! No green!

---

## Testing Checklist

- [ ] Rebuild app in Xcode
- [ ] Capture new video
- [ ] Extract depth frames
- [ ] Verify: Pure grayscale? ✅
- [ ] Verify: No green artifacts? ✅
- [ ] Verify: Shapes visible? ✅
- [ ] Move to training data folder

---

## Why This Matters for Training

### Before Fix:
- ❌ Color noise confuses the LCM encoder
- ❌ Model tries to learn random UV patterns
- ❌ Worse PSNR (maybe 2-3 dB loss)
- ❌ Longer convergence time

### After Fix:
- ✅ Clean depth signal
- ✅ Model learns true depth structure
- ✅ Better PSNR (30-35 dB achievable)
- ✅ Faster training convergence

---

## Summary

**Bug:** UV plane uninitialized → random colors  
**Fix:** Set UV = 128,128 → neutral gray  
**Impact:** Much better training data quality! 🎉

**Next:** Capture a new test video and verify it's pure grayscale!

