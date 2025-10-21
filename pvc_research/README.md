# Procedural Video Codec (PVC) Research

**Demoscene-Inspired Video Compression for Animation and Stylized Content**

---

## 🎯 **Overview**

The Procedural Video Codec (PVC) encodes videos as **programs + parameters** instead of pixels, inspired by the Demoscene's ability to generate complex audiovisual scenes from <64 KB executables.

### **Key Concept:**
- **Traditional codecs:** Store pixel data (even if compressed)
- **PVC:** Store geometric contours, motion functions, and procedural texture parameters
- **Result:** Extreme compression ratios for animation/stylized content

### **Target Content:**
✅ 2D/3D animation  
✅ Anime  
✅ Motion graphics  
✅ Game captures  
✅ Visualizations  

❌ Live-action video  
❌ High-entropy natural scenes  

---

## 📐 **Architecture**

```
┌─────────────┐
│ Input Video │
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│ Contour Extraction   │  ← OpenCV Canny + spline fitting
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Motion Tracking      │  ← Optical flow + smooth functions
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Texture Assignment   │  ← Perlin/Worley/fBM parameters
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Scene Description    │  ← JSON ISP (Intermediate Scene Program)
│  (ISP)               │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Procedural Renderer  │  ← Reconstruct frames from ISP
└──────┬───────────────┘
       │
       ▼
┌─────────────┐
│ Output Video│
└─────────────┘
```

---

## 🚀 **Quick Start**

### **Installation:**

```bash
cd pvc_research
pip install -r requirements.txt
```

### **Encoding:**

```bash
python encoder.py \
  --input anime_clip.mp4 \
  --output scene.json \
  --max-frames 300
```

### **Decoding:**

```bash
python decoder.py \
  --input scene.json \
  --output reconstructed.mp4
```

### **Evaluation:**

```bash
python decoder.py \
  --input scene.json \
  --output reconstructed.mp4 \
  --original anime_clip.mp4 \
  --evaluate
```

---

## 📊 **Performance Goals**

| Metric | Target |
|--------|--------|
| Bitrate Reduction | ≥ 90% vs AV1 |
| PSNR | ≥ 30 dB |
| SSIM | ≥ 0.85 |
| VMAF | ≥ 80 |

### **Example:**
```
Input: 10s anime @ 720p30
  AV1 baseline: ~5 Mbps = 6.25 MB
  PVC target: ≤0.5 Mbps = 625 KB
  
Compression: 90% reduction (10x smaller!)
```

---

## 🛠️ **Components**

### **Encoder Modules:**

| Module | Purpose |
|--------|---------|
| `contour_extractor.py` | Edge detection, contour vectorization |
| `motion_tracker.py` | Optical flow, motion modeling |
| `encoder.py` | Main encoding pipeline |

### **Decoder Modules:**

| Module | Purpose |
|--------|---------|
| `procedural_textures.py` | Perlin, Worley, fBM noise generation |
| `scene_renderer.py` | Frame reconstruction from ISP |
| `decoder.py` | Main decoding pipeline |

### **Utils:**

| Module | Purpose |
|--------|---------|
| `bitrate_calculator.py` | Compression ratio analysis |
| `quality_metrics.py` | PSNR, SSIM, VMAF calculation |

---

## 🎨 **Procedural Textures**

PVC supports multiple procedural texture types:

### **1. Perlin Noise**
```python
{
  "type": "perlin",
  "params": {
    "scale": 10.0,
    "octaves": 4,
    "persistence": 0.5,
    "lacunarity": 2.0,
    "seed": 42
  }
}
```

### **2. Worley (Cellular) Noise**
```python
{
  "type": "worley",
  "params": {
    "num_points": 20,
    "distance_func": "euclidean",
    "seed": 42
  }
}
```

### **3. Fractional Brownian Motion (fBM)**
```python
{
  "type": "fbm",
  "params": {
    "base_scale": 15.0,
    "octaves": 6,
    "persistence": 0.5,
    "lacunarity": 2.0,
    "seed": 42
  }
}
```

---

## 📄 **Scene Description Format**

The Intermediate Scene Program (ISP) is a JSON file:

```json
{
  "metadata": {
    "resolution": [1280, 720],
    "fps": 30,
    "frame_count": 300,
    "duration": 10.0
  },
  "objects": [
    {
      "id": 0,
      "contour_sequence": [
        {
          "points": [[x1, y1], [x2, y2], ...],
          "centroid": [cx, cy]
        }
      ],
      "motion_model": {
        "type": "spline",
        "params": { ... }
      },
      "texture": {
        "type": "perlin",
        "params": { ... }
      }
    }
  ],
  "residuals": []
}
```

---

## 🧪 **Testing**

### **Generate Test Texture Samples:**

```bash
cd decoder
python procedural_textures.py
```

### **Test Encoder on Sample:**

```bash
python encoder.py \
  --input test_clips/anime_sample.mp4 \
  --output experiments/test1.json \
  --max-frames 100
```

### **Test Decoder:**

```bash
python decoder.py \
  --input experiments/test1.json \
  --output experiments/test1_reconstructed.mp4 \
  --original test_clips/anime_sample.mp4 \
  --evaluate
```

---

## 🔬 **Research Questions**

1. **What compression ratios can we achieve on anime vs. 2D vs. 3D animation?**
2. **How does PVC compare to AV1 on different content types?**
3. **What's the trade-off between contour complexity and bitrate?**
4. **Can we use neural networks to optimize procedural parameters?**
5. **How much do residuals improve quality for complex scenes?**

---

## 📈 **Roadmap**

- [x] Core encoder/decoder pipeline
- [x] Contour extraction and tracking
- [x] Procedural texture generation
- [x] Bitrate and quality metrics
- [ ] Residual encoding for high-entropy regions
- [ ] GLSL shader optimization
- [ ] GPU-accelerated rendering
- [ ] Real-time preview
- [ ] AWS deployment and automation
- [ ] Dashboard integration
- [ ] Batch testing framework

---

## 🎓 **Inspiration**

- **Demoscene:** [Ctrl-Alt-Test](https://www.ctrl-alt-test.fr)
- **Procedural Generation:** [Book of Shaders](https://thebookofshaders.com/)
- **Video Compression:** AV1, HEVC, VVC research

---

## 📚 **Documentation**

- `PROJECT_PLAN.md` - Detailed project plan and milestones
- `V3_SYSTEM_DESIGN.md` - Integration with main framework
- `encoder/*.py` - Encoder module documentation (inline)
- `decoder/*.py` - Decoder module documentation (inline)

---

## 🤝 **Contributing**

This is an active research project. Key areas for contribution:

1. **Texture Classification:** Better heuristics for assigning procedural textures
2. **Motion Models:** More sophisticated motion fitting (affine, perspective)
3. **Residual Encoding:** Implement sparse tile encoding
4. **GPU Rendering:** GLSL shader implementation
5. **Evaluation:** More test cases and benchmarks

---

## 📧 **Contact**

**Yaron Torbaty**  
LinkedIn: [linkedin.com/in/yaron-torbaty](https://www.linkedin.com/in/yaron-torbaty/)  
Project: AiV1 Video Codec Research v3.0

---

## 📝 **License**

Research project - see main repository for license details.

---

**Status:** Active Development  
**Last Updated:** October 19, 2025  
**Version:** 1.0 (Prototype)

