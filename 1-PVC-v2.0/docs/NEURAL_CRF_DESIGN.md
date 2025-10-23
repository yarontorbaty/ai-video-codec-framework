# Neural CRF Design for PVC v2.0

## What is CRF?

In traditional codecs (H.264, HEVC, AV1):
- **CRF (Constant Rate Factor)** is a quality slider: 0 = lossless, 51 = worst
- Lower CRF = higher quality, larger files
- CRF 18-28 is typical for high-quality encodes
- CRF controls the quantization parameter dynamically

---

## Neural Codec Equivalent: Quality Levels

Our neural codec has **multiple quality knobs** we can tune:

### 1. **Latent Channels** (Primary Quality Control)
The number of channels in the compressed latent representation.

| Latent Channels | Compressed Size (960×540) | Est. PSNR | Use Case |
|----------------|---------------------------|-----------|----------|
| **16** | ~6 KB | 38-42 dB | Ultra-low bandwidth |
| **24** | ~9 KB | 42-46 dB | Mobile streaming |
| **32** | ~12.6 KB | 46-50 dB | High quality (current) |
| **48** | ~19 KB | 48-52 dB | Archival |
| **64** | ~25 KB | 50-54 dB | Near-lossless |

**Implementation:** Train multiple decoder models, one for each latent size.

---

### 2. **Quantization Bits** (Secondary Quality Control)
The bit depth used to store latent values.

| Bit Depth | Size Multiplier | Quality Loss | Use Case |
|-----------|----------------|--------------|----------|
| **INT4** | 0.25× | ~2-3 dB loss | Extreme compression |
| **INT8** | 0.5× | ~0.5-1 dB loss | Standard compression |
| **FP16** | 1.0× | Minimal loss | High quality |
| **FP32** | 2.0× | No loss | Research/training |

**Current:** Using INT8 + GZIP (0.5×)

---

### 3. **Procedural Function Count** (Tertiary Quality Control)
Number of procedural graphics functions predicted per frame.

| Function Count | Procedural Quality | Use Case |
|---------------|-------------------|----------|
| **0** | Residuals only | Simple content |
| **5** | Basic shapes | Low complexity |
| **10** | Good reconstruction | Anime/animation (current) |
| **20** | Excellent detail | Complex scenes |
| **50** | Near-perfect | Archival |

---

## Proposed "Neural CRF" Scale (0-51)

We can map our quality controls to a familiar 0-51 scale:

### **Scale Design:**

```
Neural CRF = f(latent_channels, quantization, functions)

Neural CRF 0-17:   Ultra-high quality (48+ dB PSNR)
Neural CRF 18-28:  High quality (42-48 dB PSNR) 
Neural CRF 29-39:  Medium quality (36-42 dB PSNR)
Neural CRF 40-51:  Low quality (30-36 dB PSNR)
```

### **Mapping Table:**

| Neural CRF | Latent Ch | Quant | Functions | Est. PSNR | Size (960×540) | AV1 Equivalent |
|-----------|-----------|-------|-----------|-----------|----------------|----------------|
| **0** | 64 | FP16 | 50 | 54+ dB | 50 KB | CRF 0-5 |
| **5** | 64 | INT8 | 50 | 52 dB | 25 KB | CRF 6-10 |
| **10** | 48 | INT8 | 20 | 50 dB | 19 KB | CRF 11-14 |
| **15** | 32 | INT8 | 20 | 48 dB | 12.6 KB | CRF 15-17 |
| **18** | 32 | INT8 | 10 | 46 dB | 12.6 KB | **CRF 18** (current) |
| **23** | 24 | INT8 | 10 | 44 dB | 9 KB | CRF 20-23 |
| **28** | 24 | INT8 | 5 | 42 dB | 9 KB | CRF 24-28 |
| **33** | 16 | INT8 | 5 | 40 dB | 6 KB | CRF 29-33 |
| **38** | 16 | INT8 | 0 | 38 dB | 6 KB | CRF 34-38 |
| **43** | 16 | INT4 | 0 | 36 dB | 3 KB | CRF 39-43 |
| **48** | 8 | INT4 | 0 | 34 dB | 1.5 KB | CRF 44-48 |
| **51** | 8 | INT4 | 0 | 32 dB | 1.5 KB | CRF 49-51 |

**Current model:** Neural CRF 18 (equivalent to AV1 CRF 12 in quality)

---

## Implementation Plan

### Phase 1: Multi-Model Approach (Quick, 2 weeks)

Train separate models for each quality tier:

```python
models = {
    'ultra': {'latent_ch': 64, 'functions': 50},  # Neural CRF 0-10
    'high':  {'latent_ch': 32, 'functions': 20},  # Neural CRF 11-20
    'mid':   {'latent_ch': 24, 'functions': 10},  # Neural CRF 21-30
    'low':   {'latent_ch': 16, 'functions': 5},   # Neural CRF 31-40
    'ultra_low': {'latent_ch': 8, 'functions': 0}, # Neural CRF 41-51
}

def encode(frame, neural_crf=18):
    if neural_crf <= 10:
        model = models['ultra']
        quant = 'fp16'
    elif neural_crf <= 20:
        model = models['high']
        quant = 'int8'
    elif neural_crf <= 30:
        model = models['mid']
        quant = 'int8'
    elif neural_crf <= 40:
        model = models['low']
        quant = 'int8'
    else:
        model = models['ultra_low']
        quant = 'int4'
    
    latent = model.encode(frame)
    compressed = quantize(latent, quant)
    return compressed
```

**Pros:** 
- Simple to implement
- Each model optimized for its quality tier
- Can mix and match quantization

**Cons:**
- Requires 5 trained models (250 MB total)
- Cannot fine-tune quality within tiers

---

### Phase 2: Variable Latent Model (Better, 4-6 weeks)

Train a single model that supports variable latent channels:

```python
class VariableLatentEncoder(nn.Module):
    def __init__(self, max_channels=64):
        super().__init__()
        self.encoder = ConvEncoder(out_channels=max_channels)
        self.channel_mask = None
    
    def set_quality(self, neural_crf):
        # Map CRF to channel count
        channels = {
            (0, 10): 64,
            (11, 20): 48,
            (21, 30): 32,
            (31, 40): 24,
            (41, 51): 16,
        }
        
        for (low, high), ch in channels.items():
            if low <= neural_crf <= high:
                self.latent_channels = ch
                break
        
        # Create channel mask (keep most important channels)
        self.channel_mask = torch.ones(self.latent_channels)
    
    def forward(self, x):
        latent = self.encoder(x)  # Full 64 channels
        
        # Mask to desired channel count
        if self.channel_mask is not None:
            latent = latent[:, :self.latent_channels, :, :]
        
        return latent
```

**Training strategy:**
- Train encoder to output 64 channels
- Train decoder to reconstruct from any subset of channels (16, 24, 32, 48, 64)
- Use progressive training: start with 64ch, gradually reduce

**Pros:**
- Single model (59 MB)
- Smooth quality transitions
- Runtime quality selection

**Cons:**
- More complex training
- Decoder needs to handle variable input sizes

---

### Phase 3: Learned Rate-Distortion (Best, 8-10 weeks)

Train the model to optimize for a target bitrate directly:

```python
class RateDistortionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvEncoder()
        self.rate_adapter = nn.Sequential(
            nn.Linear(1, 128),  # Input: target CRF
            nn.ReLU(),
            nn.Linear(128, 64),  # Output: channel importance weights
            nn.Sigmoid()
        )
    
    def forward(self, x, neural_crf):
        # Encode to full latent
        latent = self.encoder(x)  # (B, 64, H, W)
        
        # Get channel importance weights based on CRF
        crf_tensor = torch.tensor([neural_crf / 51.0])  # Normalize
        importance = self.rate_adapter(crf_tensor)  # (64,)
        
        # Weight channels by importance
        latent_weighted = latent * importance.view(1, 64, 1, 1)
        
        return latent_weighted, importance

class RateDistortionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = ConvDecoder()
    
    def forward(self, latent_weighted):
        # Decoder learns to handle variable importance weights
        return self.decoder(latent_weighted)
```

**Training Loss:**
```python
def train_step(model, frame, neural_crf):
    latent, importance = model.encoder(frame, neural_crf)
    reconstructed = model.decoder(latent)
    
    # Distortion loss
    mse_loss = F.mse_loss(reconstructed, frame)
    
    # Rate loss (penalize using too many channels)
    target_rate = (51 - neural_crf) / 51.0  # Higher CRF = lower rate
    actual_rate = importance.sum() / 64.0
    rate_loss = F.mse_loss(actual_rate, target_rate)
    
    # Combined loss
    loss = mse_loss + 0.01 * rate_loss
    return loss
```

**Pros:**
- True rate-distortion optimization
- Smooth quality control
- Learns optimal channel allocation per CRF
- Single model

**Cons:**
- Complex training
- Requires large dataset
- 8-10 weeks development time

---

## Recommended Implementation: Phase 1 (Multi-Model)

For quick deployment, I recommend **Phase 1**:

### Training Plan:

1. **Train 5 models** (1 week):
   - Ultra: 64ch, 50 functions
   - High: 32ch, 20 functions (current)
   - Mid: 24ch, 10 functions
   - Low: 16ch, 5 functions
   - Ultra-low: 8ch, 0 functions

2. **Benchmark each model** (2 days):
   - Test on real anime
   - Measure PSNR, SSIM, VMAF
   - Measure compressed file sizes
   - Create quality/bitrate curves

3. **Create CRF mapping** (1 day):
   - Map each model to CRF range
   - Document quality/size tradeoffs
   - Publish comparison vs AV1

### Usage Example:

```bash
# Encode with Neural CRF 18 (high quality, current default)
pvc_encode input.mp4 output.pvc --neural-crf 18

# Encode with Neural CRF 28 (smaller files)
pvc_encode input.mp4 output.pvc --neural-crf 28

# Encode with Neural CRF 10 (archival quality)
pvc_encode input.mp4 output.pvc --neural-crf 10
```

---

## Quality/Bitrate Comparison

### Projected Results:

| Neural CRF | PSNR | Size (1080p) | Bitrate @ 30fps | vs AV1 CRF 30 |
|-----------|------|--------------|-----------------|---------------|
| **10** | 50 dB | 76 KB/frame | 18.2 Mbps | +7 dB, 1.13× size |
| **15** | 48 dB | 50 KB/frame | 12.1 Mbps | +5 dB, 0.74× size |
| **18** | 46 dB | 50 KB/frame | 12.1 Mbps | +3 dB, 0.74× size |
| **23** | 44 dB | 36 KB/frame | 8.6 Mbps | +1 dB, 0.53× size |
| **28** | 42 dB | 36 KB/frame | 8.6 Mbps | -1 dB, 0.53× size |
| **33** | 40 dB | 24 KB/frame | 5.8 Mbps | -3 dB, 0.36× size |

**AV1 CRF 30 baseline:** 43 dB, 67 KB/frame, 16.1 Mbps @ 1080p

---

## Implementation Checklist

- [ ] Design multi-model architecture (different latent sizes)
- [ ] Generate training datasets for each quality tier
- [ ] Train 5 models (Ultra, High, Mid, Low, Ultra-low)
- [ ] Benchmark all models on real anime test set
- [ ] Create CRF-to-model mapping table
- [ ] Implement encoder with `--neural-crf` flag
- [ ] Document quality/bitrate curves
- [ ] Publish comparison vs AV1 at matched CRF levels

**Timeline:** 2-3 weeks  
**Cost:** ~$50-100 (GPU training for 5 models)  
**Result:** Production-ready quality control system

---

## Future Enhancements

1. **Adaptive CRF:** Adjust quality per-frame based on scene complexity
2. **Two-pass encoding:** Analyze entire video, then optimize CRF per scene
3. **Region-based CRF:** Higher quality for faces, lower for backgrounds
4. **Perceptual CRF:** Optimize for VMAF instead of PSNR
5. **Neural rate control:** Train model to hit exact target bitrate

---

**Conclusion:** Neural CRF is highly feasible and maps well to traditional CRF. Phase 1 (multi-model) can be implemented in 2-3 weeks and provides AV1-equivalent quality control.
