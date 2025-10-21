# PVC v2.0 - Procedural Video Codec

**Neural-Procedural Hybrid for Animation/Anime Compression**

---

## 📁 What's in This Folder

This folder contains the complete PVC v2.0 project:

### `pvc_v2/` - Main Codebase
- **graphics/** - 47 graphics primitives library
- **models/** - PVC + SOTA residual codecs (32M params)
- **training/** - Training scripts (quick, full, hybrid)
- **tests/** - Evaluation tools and demo generators

### `pvc_research/` - Early Research
- Initial proof-of-concept code
- PVC v1.0 experiments
- Procedural-only approaches

### `docs/` - Documentation
- **architecture/** - Technical designs
- **reports/** - Training results & benchmarks
- **AI_CODEC_EVOLUTION_ROADMAP.md** - Complete project history

---

## 🎯 Current Status

**Latest Results:** 23.82 dB PSNR (quick test)  
**Training:** Epoch 16/50 (full model) 🔄  
**Target:** 30-40 dB for production

---

## 🚀 Quick Start

```bash
cd pvc_v2

# Install dependencies
pip install -r requirements.txt

# Evaluate trained model
python3 tests/eval_sota.py

# Generate comparison video
python3 tests/create_comparison_video.py --frames 100 --fps 30
```

---

## 📊 Key Results

| Model | PSNR | Status |
|-------|------|--------|
| Baseline | 11.22 dB | ✅ |
| Simple Hybrid | 19.91 dB | ✅ |
| SOTA Quick | 23.82 dB | ✅ |
| SOTA Full | 25-28 dB | 🔄 Training |

---

**See main repository README for full documentation**

