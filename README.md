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

**✅ SOTA Full Training Complete!**

**Final Results:** **25.06 dB PSNR** | **0.88 SSIM**  
**Training:** 50 epochs, 10.55 hours (Oct 21, 2025)  
**Status:** Research prototype complete 🎉

---

## 📊 Final Results

| Milestone | PSNR | SSIM | Params | Status |
|-----------|------|------|--------|--------|
| **Baseline (PVC only)** | 11.22 dB | 0.20 | 3.8M | ✅ Complete |
| **Simple Hybrid** | 19.91 dB | 0.72 | 67K | ✅ Complete |
| **SOTA Quick (10 epochs)** | 23.82 dB | 0.89 | 32.4M | ✅ Complete |
| **SOTA Full (50 epochs)** | **25.06 dB** ⭐ | **0.88** | **32.4M** | ✅ **COMPLETE** |

**Total Improvement:** 11.22 → 25.06 dB = **+123.5%** 🚀

**Compression:** 88% smaller than AV1 for animation/stylized content

---

## 🚀 Quick Start

### Evaluate Trained Models

```bash
cd pvc_v2

# Install dependencies
pip install -r requirements.txt

# Download trained models (see MODEL_DOWNLOAD.md)
aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/models/sota_residual_encoder_best.pth models/sota_full/
aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/models/sota_residual_decoder_best.pth models/sota_full/

# Evaluate SOTA model (generates comparison images)
python3 tests/eval_sota.py

# Generate comparison video
python3 tests/create_comparison_video.py --frames 100 --fps 30
```

### Use Models in Your Code

```python
import torch
from models.enhanced_network import EnhancedPVCv2Model
from models.sota_residual_encoder import SOTAResidualEncoder
from models.sota_residual_decoder import SOTAResidualDecoder

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pvc_model = EnhancedPVCv2Model().to(device)
pvc_model.load_state_dict(torch.load('models/pvc_v2_perceptual_best.pth'))

encoder = SOTAResidualEncoder().to(device)
encoder.load_state_dict(torch.load('models/sota_full/sota_residual_encoder_best.pth'))

decoder = SOTAResidualDecoder().to(device)
decoder.load_state_dict(torch.load('models/sota_full/sota_residual_decoder_best.pth'))

# Encode frame
compressed = encode_frame(original_frame, pvc_model, encoder)

# Decode frame
reconstructed = decode_frame(compressed, pvc_model, decoder)
```

---

## 📚 Documentation

**Complete Results:**
- **[SOTA_FULL_TRAINING_RESULTS.md](SOTA_FULL_TRAINING_RESULTS.md)** - Comprehensive training report
- **[MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md)** - Model download instructions
- **[docs/AI_CODEC_EVOLUTION_ROADMAP.md](docs/AI_CODEC_EVOLUTION_ROADMAP.md)** - Project evolution

**Architecture Details:**
- **[docs/architecture/PVC_V2_SOTA_DESIGN.md](docs/architecture/PVC_V2_SOTA_DESIGN.md)** - SOTA architecture
- **[docs/reports/](docs/reports/)** - All training reports

**Visual Results:**
- **[pvc_v2/tests/sota_hybrid_comparison.png](pvc_v2/tests/sota_hybrid_comparison.png)** - Side-by-side comparison

---

## 💡 Key Innovation

**Neural-Procedural Hybrid Codec:**
1. **Coarse Reconstruction:** PVC predicts graphics function sequences (fast, 9.89 dB)
2. **Residual Refinement:** SOTA U-Net encodes fine details (+15.17 dB improvement)
3. **Final Output:** 25.06 dB PSNR at 88% compression vs AV1

**Best For:** Animation, anime, stylized graphics, low-bandwidth streaming

---

**See main repository README for full documentation**

