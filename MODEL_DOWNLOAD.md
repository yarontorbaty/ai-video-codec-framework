# PVC v2.0 - Trained Model Files

## 📥 Public Model Downloads

The trained model files are **publicly available** for download:

---

## 🔥 **Latest: Production Models (Epoch 39/100)**

**Status:** Training in progress, current best models available

### Direct Download Links (No AWS Account Required):

**Option 1: Browser Download (Latest)**
- [Production Encoder - Epoch 39 (252 MB)](https://ai-codec-v3-artifacts-580473065386.s3.us-east-1.amazonaws.com/pvc/models/production_encoder_best_epoch39.pth)
- [Production Decoder - Epoch 39 (104 MB)](https://ai-codec-v3-artifacts-580473065386.s3.us-east-1.amazonaws.com/pvc/models/production_decoder_best_epoch39.pth)

**Option 2: Command Line (Latest)**

```bash
# Using wget
wget https://ai-codec-v3-artifacts-580473065386.s3.us-east-1.amazonaws.com/pvc/models/production_encoder_best_epoch39.pth
wget https://ai-codec-v3-artifacts-580473065386.s3.us-east-1.amazonaws.com/pvc/models/production_decoder_best_epoch39.pth

# Or using curl
curl -O https://ai-codec-v3-artifacts-580473065386.s3.us-east-1.amazonaws.com/pvc/models/production_encoder_best_epoch39.pth
curl -O https://ai-codec-v3-artifacts-580473065386.s3.us-east-1.amazonaws.com/pvc/models/production_decoder_best_epoch39.pth
```

**Option 3: AWS CLI**

```bash
aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/models/production_encoder_best_epoch39.pth .
aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/models/production_decoder_best_epoch39.pth .
```

### Latest Model Details:

**Production Residual Encoder (Epoch 39):**
- File: `production_encoder_best_epoch39.pth`
- Size: 252 MB
- Parameters: 65,930,000 (65.9M)
- Architecture: Enhanced U-Net with attention (5 encoding blocks, up to 640 channels)

**Production Residual Decoder (Epoch 39):**
- File: `production_decoder_best_epoch39.pth`
- Size: 104 MB
- Parameters: 27,000,000 (27.0M)
- Architecture: Enhanced U-Net decoder with skip connections (5 upsampling blocks)

**Total:** 92,930,000 parameters (93M), 356 MB

### Latest Results (Epoch 39):

- **PSNR:** 26.36 dB (real anime frame test)
- **SSIM:** 0.8322
- **Compression:** 1.9 KB per 256×256 frame (INT8 + GZIP)
- **vs JPEG Q7:** 1.28× smaller at matched PSNR, 7.4% better SSIM
- **Training:** 4× A10G GPUs (g5.12xlarge), ongoing

---

## 📦 **Previous: SOTA Models (Baseline)**

### Direct Download Links:

**Option 1: Browser Download (Baseline)**
- [SOTA Residual Encoder (77 MB)](https://ai-codec-v3-artifacts-580473065386.s3.us-east-1.amazonaws.com/pvc/models/sota_residual_encoder_best.pth)
- [SOTA Residual Decoder (47 MB)](https://ai-codec-v3-artifacts-580473065386.s3.us-east-1.amazonaws.com/pvc/models/sota_residual_decoder_best.pth)

**Option 2: Command Line (Baseline)**

```bash
# Using wget
wget https://ai-codec-v3-artifacts-580473065386.s3.us-east-1.amazonaws.com/pvc/models/sota_residual_encoder_best.pth
wget https://ai-codec-v3-artifacts-580473065386.s3.us-east-1.amazonaws.com/pvc/models/sota_residual_decoder_best.pth

# Or using curl
curl -O https://ai-codec-v3-artifacts-580473065386.s3.us-east-1.amazonaws.com/pvc/models/sota_residual_encoder_best.pth
curl -O https://ai-codec-v3-artifacts-580473065386.s3.us-east-1.amazonaws.com/pvc/models/sota_residual_decoder_best.pth
```

**Option 3: AWS CLI (Baseline)**

```bash
aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/models/sota_residual_encoder_best.pth .
aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/models/sota_residual_decoder_best.pth .
```

### Baseline Model Details:

**SOTA Residual Encoder:**
- File: `sota_residual_encoder_best.pth`
- Size: 77 MB (76.9 MiB)
- Parameters: 20,171,337 (20.2M)
- Architecture: U-Net with attention blocks

**SOTA Residual Decoder:**
- File: `sota_residual_decoder_best.pth`
- Size: 47 MB (46.6 MiB)
- Parameters: 12,225,059 (12.2M)
- Architecture: U-Net decoder with skip connections

**Total:** 32,396,396 parameters (32.4M), 124 MB

### Baseline Results:

- **PSNR:** 25.06 dB
- **SSIM:** 0.88
- **Training:** 50 epochs, 10.55 hours (g4dn.xlarge)

### Why Not in Git?

These model files are too large for GitHub (124 MB total). While each individual file is under GitHub's 100 MB limit, they cause connection issues during push. 

**Alternative:** These models are hosted in AWS S3 for easy access.

### Usage:

```python
import torch
from models.sota_residual_encoder import SOTAResidualEncoder
from models.sota_residual_decoder import SOTAResidualDecoder

# Load models
encoder = SOTAResidualEncoder().to('cuda')
encoder.load_state_dict(torch.load('sota_residual_encoder_best.pth'))

decoder = SOTAResidualDecoder().to('cuda')
decoder.load_state_dict(torch.load('sota_residual_decoder_best.pth'))

# Use for inference...
```

### Training Results:

- **PSNR:** 25.06 ± 11.24 dB
- **SSIM:** 0.8834 ± 0.0931
- **Training:** 50 epochs, 10.55 hours
- **Hardware:** AWS g4dn.xlarge (Tesla T4)

See `SOTA_FULL_TRAINING_RESULTS.md` for complete training details and results.

---

**Status:** Production-ready models for animation/stylized content compression

