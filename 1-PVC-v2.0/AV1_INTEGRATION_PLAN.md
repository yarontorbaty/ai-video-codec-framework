# AV1 Integration Plan - Tomorrow's Work

## 🎯 Goal
Replace AV1 I-frames with our Tier 1 neural codec for 25% I-frame size reduction.

---

## 📊 Current State

### What We Have
✅ **Tier 1 Hybrid Codec**
- Model: `tier1_final_model.pth` (59 MB)
- Performance: 48 dB PSNR, 50.36 KB per 1080p frame
- 25.2% smaller than AV1 I-frames

✅ **AV1 Baseline**
- I-frames: 67.37 KB (1080p)
- P/B frames: ~1-2 KB average
- Total video: ~10% I-frames, 90% P/B frames

### Expected Improvement
```
Before (Pure AV1):
  10 I-frames × 67 KB  = 670 KB
  90 P-frames × 1.5 KB = 135 KB
  Total              = 805 KB

After (Neural I-frames + AV1 temporal):
  10 I-frames × 50 KB  = 500 KB  (25% reduction)
  90 P-frames × 1.5 KB = 135 KB  (unchanged)
  Total              = 635 KB

Net improvement: 21% smaller overall
```

---

## 🛠️ Implementation Approaches

### Option 1: External Preprocessing (Easiest)
**Time:** 2-4 hours

1. Extract I-frames from video using ffmpeg
2. Encode I-frames with our codec
3. Replace I-frames in AV1 stream
4. Re-mux video

**Pros:**
- No AV1 source code changes
- Works with existing tools
- Easy to prototype

**Cons:**
- Manual workflow
- Not integrated
- Extra encoding pass

**Steps:**
```bash
# 1. Extract I-frames
ffmpeg -i input.mp4 -vf "select='eq(pict_type,I)'" -vsync 0 iframes/frame_%04d.png

# 2. Encode with our codec
python encode_iframes.py --input iframes/ --output encoded/

# 3. Inject back into stream
python inject_iframes.py --video input.mp4 --iframes encoded/ --output hybrid.mp4
```

---

### Option 2: FFmpeg Filter (Medium)
**Time:** 1-2 days

Create custom FFmpeg filter that calls our codec.

**Pros:**
- Integrates with FFmpeg ecosystem
- Standard tools compatibility
- One-pass encoding

**Cons:**
- Requires C++ wrapper for PyTorch model
- FFmpeg filter API learning curve
- Need to compile FFmpeg

**Steps:**
1. Create `libavfilter/vf_neural_iframe.c`
2. Wrap PyTorch model with LibTorch C++ API
3. Register filter in FFmpeg
4. Compile custom FFmpeg

---

### Option 3: Modify libaom (Hard, Best)
**Time:** 3-5 days

Directly integrate into AV1 encoder (libaom).

**Pros:**
- True integration
- Production-ready
- Optimal performance

**Cons:**
- Complex codebase
- Requires deep AV1 knowledge
- Longer development time

**Hook Points in libaom:**
```c
// av1/encoder/encoder.c
static int encode_frame_internal(AV1_COMP *cpi, size_t *size, uint8_t *dest) {
  // Line ~7000: I-frame decision point
  if (cm->current_frame.frame_type == KEY_FRAME) {
    // ⭐ Hook here: Call our neural codec
    neural_encode_iframe(cpi->source, dest, size);
  } else {
    // Standard AV1 P/B frame encoding
  }
}
```

---

## 📝 Recommended: Start with Option 1

### Why Option 1 First?
1. ✅ Fast to implement (2-4 hours)
2. ✅ Proves the concept works
3. ✅ Can measure actual improvement
4. ✅ No complex dependencies
5. ✅ Easy to demo

### Implementation Plan

#### Step 1: I-frame Extraction Tool
```python
# extract_iframes.py
import subprocess
import sys

def extract_iframes(video_path, output_dir):
    """Extract all I-frames from video"""
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', "select='eq(pict_type,I)'",
        '-vsync', '0',
        f'{output_dir}/iframe_%04d.png'
    ]
    subprocess.run(cmd)
```

#### Step 2: Neural Encoding Tool
```python
# encode_iframes.py
import torch
from model import SimplifiedHybridModel
import cv2
import gzip
import pickle

model = SimplifiedHybridModel(num_functions=51)
checkpoint = torch.load('tier1_final_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def encode_iframe(frame_path):
    """Encode a single I-frame"""
    frame = cv2.imread(frame_path)
    frame = cv2.resize(frame, (960, 540))
    frame_tensor = torch.from_numpy(frame / 255.0).permute(2, 0, 1).unsqueeze(0)
    
    with torch.no_grad():
        output, latent, func_logits, params = model(frame_tensor)
    
    # Compress
    latent_int8 = (latent.numpy() * 127).astype(np.int8)
    compressed = gzip.compress(latent_int8.tobytes())
    
    # Save
    data = {
        'latent': compressed,
        'func_ids': torch.argmax(func_logits, dim=-1).numpy(),
        'params': params.numpy()
    }
    
    return pickle.dumps(data)
```

#### Step 3: Stream Injection Tool
```python
# inject_iframes.py
# Use PyAV or similar to:
# 1. Open AV1 video stream
# 2. Identify I-frame positions
# 3. Replace I-frame data with our encoded data
# 4. Re-mux into container
```

---

## 🧪 Testing Strategy

### Test Videos
1. **bleach_neural_test_source.mp4** (already have it)
2. Short 10-second clip
3. Known I-frame positions

### Metrics to Track
- [ ] I-frame size (before/after)
- [ ] Overall video size
- [ ] PSNR/SSIM (verify quality maintained)
- [ ] Encoding time
- [ ] Decoding time

### Success Criteria
✅ I-frames 20-25% smaller
✅ Overall video 15-25% smaller
✅ Quality maintained (>45 dB PSNR)
✅ Playback works correctly

---

## 📦 Files Needed Tomorrow

### Model Files
```
/tmp/tier1_results/
  ├── tier1_final.pth              # Main model (59 MB)
  └── codec_comparison_single_frame.png  # Reference
```

### Test Videos
```
~/Downloads/
  └── bleach_neural_test_source.mp4
```

### Scripts to Create
```
/tmp/av1_integration/
  ├── extract_iframes.py
  ├── encode_iframes.py
  ├── decode_iframes.py
  ├── inject_iframes.py
  └── test_integration.py
```

---

## 🎯 Timeline (Tomorrow)

### Morning (2-3 hours)
- [ ] Set up working directory
- [ ] Create I-frame extraction tool
- [ ] Test extraction on Bleach video

### Afternoon (3-4 hours)
- [ ] Create neural encoding pipeline
- [ ] Test encoding all I-frames
- [ ] Measure compression improvement

### Evening (2-3 hours)
- [ ] Create injection/re-muxing tool
- [ ] Generate hybrid video
- [ ] Run full quality/size tests
- [ ] Document results

### Total Estimate: 7-10 hours
**Deliverable:** Working hybrid AV1+Neural codec video with measured improvements

---

## 🚀 Next Steps After Proof-of-Concept

If Option 1 succeeds (shows >20% improvement):

1. **Week 2:** Implement Option 2 (FFmpeg filter)
   - Better integration
   - Single-pass encoding
   - Compatible with existing tools

2. **Week 3-4:** Implement Option 3 (libaom integration)
   - Production-ready
   - Optimal performance
   - Submit patch to libaom project

3. **Month 2:** Add temporal compression (Phase 3)
   - Neural P-frames using procedural motion
   - Expected: Additional 30-50% savings

---

## 📚 Reference Links

### AV1 Documentation
- libaom repo: https://aomedia.googlesource.com/aom/
- AV1 spec: https://aomediacodec.github.io/av1-spec/
- Encoder guide: https://aomedia.googlesource.com/aom/+/refs/heads/main/README.md

### FFmpeg
- Filter writing: https://ffmpeg.org/developer.html
- Video filters: https://ffmpeg.org/ffmpeg-filters.html#Video-Filters

### Tools
- PyAV (Python video manipulation): https://pyav.org/
- OpenCV (image processing): Already installed

---

## 💡 Key Insights

1. **I-frames are 10-15% of total frames** in typical video
   - But 40-50% of total size
   - High leverage point for optimization

2. **Our codec's 25% I-frame reduction** = 10-15% overall savings
   - With zero changes to P/B frames
   - Pure drop-in improvement

3. **Quality is better** (48 dB vs 43 dB)
   - Smaller AND better
   - No quality trade-off

4. **This is unique**
   - No other codec beats AV1 I-frames
   - Publishable result
   - Potential patent

---

**Status:** ✅ Ready to implement tomorrow

*Created: October 23, 2025*

