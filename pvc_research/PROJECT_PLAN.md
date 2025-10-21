# Procedural Video Codec (PVC) Research - Project Plan

**Date:** October 19, 2025  
**Status:** Planning Phase  
**Goal:** 90% bitrate reduction vs AV1 for animation/stylized content

---

## 🎯 Project Overview

### **Concept:**
Encode animation/stylized videos as **programs + parameters** instead of pixels.
- Inspired by Demoscene (64KB executables generating complex scenes)
- Target: Animation, 2D/3D graphics, motion graphics, game captures
- NOT for: Live-action, natural scenes with high entropy

### **Target Performance:**
- ≥90% bitrate reduction vs libaom-AV1
- Matched perceptual quality (VMAF)
- Resolution: ≤720p for prototype

---

## 📐 Architecture

### **Pipeline Stages:**

```
Input Video (Animation/Stylized)
         ↓
[1. Edge & Contour Extraction]
    → OpenCV Canny edge detection
    → Contour finding & grouping
    → Spline/polyline fitting
         ↓
[2. Object Motion Modeling]
    → Optical flow tracking
    → Motion vector computation
    → Smooth motion function fitting
         ↓
[3. Procedural Texture Assignment]
    → Perlin/Worley/Simplex noise
    → Texture parameters + seeds
    → Material assignment per object
         ↓
[4. Scene Program Synthesis]
    → Generate ISP (Intermediate Scene Program)
    → GLSL-like DSL output
    → Shader minification
         ↓
[5. Residual Correction (optional)]
    → Compute error map
    → Sparse tile encoding (AV1 or VQ)
    → Only where needed
         ↓
Output: Compact Scene Description
    → JSON/DSL + GLSL shaders
    → Motion parameters
    → Texture seeds
```

---

## 🛠️ Implementation Plan

### **Phase 1: Core Pipeline (Week 1)**

**Encoder Components:**
1. `contour_extractor.py`
   - Canny edge detection
   - Contour finding & grouping
   - Spline fitting
   
2. `motion_tracker.py`
   - Optical flow (Lucas-Kanade or Farneback)
   - Per-object motion vectors
   - Temporal smoothing
   
3. `texture_analyzer.py`
   - Extract texture patches
   - Classify as procedural-friendly
   - Generate procedural parameters

4. `scene_generator.py`
   - Build scene graph
   - Generate ISP JSON
   - Parameter compression

**Decoder Components:**
1. `scene_renderer.py`
   - Parse ISP
   - Apply motion transforms
   - Render procedural textures
   
2. `procedural_textures.py`
   - Perlin noise
   - Worley (cellular) noise
   - fBM (fractional Brownian motion)
   - Shader composition

**Shaders:**
1. `textures.glsl`
   - Procedural noise functions
   - Material definitions
   - Lighting models

### **Phase 2: Optimization (Week 2)**

1. **Bitrate Analysis**
   - Count bits for each component
   - Compare to AV1 baseline
   - Identify bottlenecks

2. **Quality Metrics**
   - VMAF comparison
   - PSNR/SSIM fallback
   - Perceptual error maps

3. **Residual Encoding**
   - Error detection
   - Sparse tile extraction
   - AV1 encoding for residuals

### **Phase 3: Evaluation (Week 3)**

1. **Test Suite**
   - Anime clips
   - 2D/3D animation
   - Motion graphics
   - Game captures

2. **Benchmarking**
   - PVC bitrate
   - AV1 bitrate (libaom)
   - VMAF scores
   - Encode/decode speed

3. **Documentation**
   - Design rationale
   - Usage examples
   - Performance report
   - Future improvements

---

## 📊 Success Metrics

### **Primary:**
- ✅ Bitrate reduction ≥ 90% vs AV1
- ✅ VMAF score within 5 points of original

### **Secondary:**
- Encode time < 10x real-time
- Decode time < 1x real-time
- Works on 720p content
- Modular, extensible code

### **Validation:**
```
Test Clip: 10s anime scene @ 720p30
AV1 baseline: ~5 Mbps = 6.25 MB
PVC target: ≤0.5 Mbps = 625 KB (90% reduction)
VMAF: ≥85 (vs original)
```

---

## 🔧 Technical Stack

### **Dependencies:**
```python
# Core
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.11.0

# Graphics
PyOpenGL>=3.1.6
moderngl>=5.8.0  # Or Pillow for software rendering

# Video I/O
ffmpeg-python>=0.2.0
av>=10.0.0  # PyAV for direct video access

# Evaluation
scikit-image>=0.21.0  # For SSIM/PSNR
# pyvmaf (optional, can use FFmpeg VMAF)

# Utilities
matplotlib>=3.7.0
tqdm>=4.65.0
```

### **File Structure:**
```
pvc_research/
├── encoder/
│   ├── contour_extractor.py
│   ├── motion_tracker.py
│   ├── texture_analyzer.py
│   └── scene_generator.py
├── decoder/
│   ├── scene_renderer.py
│   └── procedural_textures.py
├── shaders/
│   ├── textures.glsl
│   ├── perlin.glsl
│   └── worley.glsl
├── utils/
│   ├── bitrate_calculator.py
│   ├── quality_metrics.py
│   └── video_io.py
├── experiments/
│   ├── test_clips/
│   └── results/
├── docs/
│   ├── README.md
│   └── REPORT.md
├── encoder.py  # Main entry point
├── decoder.py  # Main entry point
└── requirements.txt
```

---

## 🎬 Example Workflow

### **Encoding:**
```bash
python encoder.py \
  --input anime_clip.mp4 \
  --output scene.json \
  --max-error 0.05 \
  --enable-residuals
```

**Output:**
```json
{
  "metadata": {
    "resolution": [1280, 720],
    "fps": 30,
    "duration": 10.0
  },
  "objects": [
    {
      "id": 0,
      "contours": [...],
      "motion": {
        "type": "affine",
        "keyframes": [...]
      },
      "texture": {
        "type": "perlin",
        "seed": 42,
        "params": {...}
      }
    }
  ],
  "residuals": [...]
}
```

### **Decoding:**
```bash
python decoder.py \
  --input scene.json \
  --output reconstructed.mp4 \
  --quality high
```

### **Evaluation:**
```bash
python evaluate.py \
  --original anime_clip.mp4 \
  --reconstructed reconstructed.mp4 \
  --report results.json
```

---

## 🧪 Test Cases

### **Test 1: Simple 2D Animation**
- **Content:** Flat colors, clear edges, simple motion
- **Expected:** 95%+ bitrate reduction
- **Challenge:** Clean contours, minimal residuals

### **Test 2: Complex Anime Scene**
- **Content:** Multiple characters, backgrounds, effects
- **Expected:** 90% bitrate reduction
- **Challenge:** Overlapping objects, motion blur

### **Test 3: 3D CGI Animation**
- **Content:** Rendered 3D with lighting
- **Expected:** 85-90% bitrate reduction
- **Challenge:** Smooth gradients, specular highlights

### **Test 4: Motion Graphics**
- **Content:** Text, shapes, transitions
- **Expected:** 95%+ bitrate reduction
- **Challenge:** Sharp edges, geometric primitives

---

## 🚀 Deployment Strategy

### **Phase 1: Local Development**
- Develop on local machine
- Test with sample clips
- Iterate on pipeline

### **Phase 2: AWS Worker Instance**
- Launch dedicated EC2 instance
- GPU-enabled for GLSL rendering
- Automated experiments

### **Phase 3: Integration**
- Separate DynamoDB table
- Dedicated dashboard
- Compare with neural codec results

---

## 📝 Deliverables

### **Code:**
- [ ] `encoder.py` - Full encoder pipeline
- [ ] `decoder.py` - Full decoder pipeline
- [ ] `textures.glsl` - Procedural shader library
- [ ] All supporting modules
- [ ] Unit tests

### **Documentation:**
- [ ] `README.md` - Setup and usage
- [ ] `REPORT.md` - Performance analysis
- [ ] `DESIGN.md` - Architecture details
- [ ] Code comments and docstrings

### **Results:**
- [ ] Bitrate comparison table
- [ ] VMAF scores
- [ ] Visual comparisons
- [ ] Performance benchmarks

---

## 🎯 Current Status

**Phase:** Planning & Setup  
**Next Steps:**
1. Create initial encoder skeleton
2. Implement contour extraction
3. Test on sample anime clip
4. Iterate on pipeline

**Timeline:**
- Week 1: Core pipeline implementation
- Week 2: Optimization and residuals
- Week 3: Evaluation and documentation

---

## 💡 Key Innovations

1. **Demoscene-Inspired Compression**
   - Programs instead of pixels
   - Procedural generation
   - Extreme compression ratios

2. **Hybrid Approach**
   - Procedural for clean regions
   - Residuals for high-entropy areas
   - Best of both worlds

3. **Content-Specific Optimization**
   - Optimized for animation/stylized
   - Not trying to be universal
   - Deep specialization = better results

---

**Ready to begin implementation!**

