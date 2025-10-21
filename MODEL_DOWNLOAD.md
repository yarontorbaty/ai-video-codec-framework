# PVC v2.0 - Trained Model Files

## Model Downloads

The trained SOTA model files are available via AWS S3:

### Download Links:

```bash
# Download encoder (77 MB)
aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/models/sota_residual_encoder_best.pth .

# Download decoder (47 MB)
aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/models/sota_residual_decoder_best.pth .
```

### Model Details:

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

