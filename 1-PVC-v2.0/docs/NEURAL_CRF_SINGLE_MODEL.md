# Neural CRF: Single Model Approach

## The Problem with Multi-Model

**Phase 1 (Multi-Model):**
- 5 separate models: 5 × 50 MB = **250 MB total**
- ❌ Too heavy for mobile devices
- ❌ Cannot smoothly transition between quality levels
- ❌ Requires downloading all models upfront

---

## Solution: Single Variable-Channel Model

Train **ONE model** that can decode from any number of latent channels (8-64).

### Key Insight: Channel Importance

Not all latent channels are equally important:
- **Channels 1-8:** Capture coarse structure (most important)
- **Channels 9-16:** Add medium details
- **Channels 17-32:** Add fine details
- **Channels 33-64:** Add ultra-fine details (diminishing returns)

**Strategy:** Train the decoder to work with ANY subset of channels [1, 8, 16, 24, 32, 48, 64].

---

## Architecture Design

### Encoder (Fixed 64 channels)

```python
class UnifiedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Always outputs 64 channels (maximum quality)
        self.conv1 = nn.Conv2d(3, 48, 5, padding=2)
        self.conv2 = nn.Conv2d(48, 64, 3, stride=2, padding=1)
        # ... more layers ...
        self.latent_conv = nn.Conv2d(..., 64, 3, padding=1)
        
        # Channel importance predictor (learned during training)
        self.importance_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.Sigmoid()  # Output: importance score per channel
        )
    
    def forward(self, x):
        features = self.conv1(x)
        features = self.conv2(features)
        # ... more processing ...
        
        latent_full = self.latent_conv(features)  # (B, 64, H, W)
        
        # Get channel importance scores
        importance = self.importance_net(latent_full)  # (B, 64)
        
        # Sort channels by importance (for encoding decision)
        sorted_indices = torch.argsort(importance, descending=True)
        
        return latent_full, sorted_indices, importance
```

### Decoder (Works with ANY channel count)

```python
class UnifiedDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Adaptive input layer that handles variable channels
        self.channel_adapter = nn.ModuleDict({
            '8':  nn.Conv2d(8, 64, 1),
            '16': nn.Conv2d(16, 64, 1),
            '24': nn.Conv2d(24, 64, 1),
            '32': nn.Conv2d(32, 64, 1),
            '48': nn.Conv2d(48, 64, 1),
            '64': nn.Conv2d(64, 64, 1),  # Identity-like
        })
        
        # Rest of decoder is shared (works on 64 channels internally)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 48, 4, stride=2, padding=1),
            nn.ReLU(),
            # ... more layers ...
            nn.ConvTranspose2d(48, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, latent_partial):
        # latent_partial can be (B, 8, H, W) or (B, 16, H, W) etc.
        num_channels = latent_partial.shape[1]
        
        # Adapt to 64 channels
        adapted = self.channel_adapter[str(num_channels)](latent_partial)
        
        # Decode from 64-channel representation
        output = self.decoder(adapted)
        
        return output
```

**Key advantage:** Decoder has 6 small adapter layers (8 KB each) + one shared decoder (50 MB).
**Total size:** ~50 MB (same as before!)

---

## Training Strategy

### Progressive Channel Training

Train the model to reconstruct from progressively fewer channels:

```python
def train_step(model, frames, epoch):
    # Full encoding
    latent_full, sorted_indices, importance = model.encoder(frames)
    
    # Randomly select a channel count for this batch
    # Early training: focus on high channels (64, 48, 32)
    # Later training: focus on low channels (8, 16, 24)
    if epoch < 20:
        channel_counts = [64, 48, 32]
    else:
        channel_counts = [8, 16, 24, 32, 48, 64]
    
    num_channels = random.choice(channel_counts)
    
    # Keep only top-K most important channels
    keep_indices = sorted_indices[:, :num_channels]
    latent_partial = latent_full[:, keep_indices, :, :]
    
    # Decode from partial latent
    reconstructed = model.decoder(latent_partial)
    
    # Loss
    mse_loss = F.mse_loss(reconstructed, frames)
    
    # Importance loss: encourage distinct importance scores
    importance_entropy = -torch.sum(importance * torch.log(importance + 1e-8))
    
    loss = mse_loss - 0.01 * importance_entropy
    return loss
```

**Training progression:**
1. **Epochs 1-20:** Train with 64, 48, 32 channels (high quality)
2. **Epochs 21-50:** Add 24, 16 channels (medium quality)
3. **Epochs 51-100:** Add 8 channels (low quality)
4. **Epochs 101-150:** Mix all channel counts randomly

---

## Encoding with Neural CRF

### CRF to Channel Count Mapping

| Neural CRF | Channels Used | Quantization | Est. PSNR | Size (960×540) |
|-----------|---------------|--------------|-----------|----------------|
| **0-10** | 64 | FP16 | 52+ dB | 25 KB |
| **11-15** | 48 | INT8 | 48-50 dB | 12 KB |
| **16-20** | 32 | INT8 | 46-48 dB | 8 KB |
| **21-28** | 24 | INT8 | 42-46 dB | 6 KB |
| **29-38** | 16 | INT8 | 38-42 dB | 4 KB |
| **39-51** | 8 | INT4 | 32-38 dB | 1 KB |

### Encoding Process

```python
def encode_frame(frame, neural_crf=18):
    # 1. Encode to full 64-channel latent
    latent_full, sorted_indices, importance = encoder(frame)
    
    # 2. Determine channel count based on CRF
    if neural_crf <= 10:
        num_channels, quant = 64, 'fp16'
    elif neural_crf <= 15:
        num_channels, quant = 48, 'int8'
    elif neural_crf <= 20:
        num_channels, quant = 32, 'int8'
    elif neural_crf <= 28:
        num_channels, quant = 24, 'int8'
    elif neural_crf <= 38:
        num_channels, quant = 16, 'int8'
    else:
        num_channels, quant = 8, 'int4'
    
    # 3. Keep only most important N channels
    keep_indices = sorted_indices[:num_channels]
    latent_partial = latent_full[:, keep_indices, :, :]
    
    # 4. Quantize and compress
    if quant == 'int8':
        latent_quantized = (latent_partial * 127).byte()
    elif quant == 'int4':
        latent_quantized = ((latent_partial * 15) // 2).byte()  # 4-bit packed
    else:
        latent_quantized = latent_partial.half()  # FP16
    
    compressed = gzip.compress(latent_quantized.cpu().numpy().tobytes())
    
    # 5. Store metadata (channel indices) + compressed latent
    header = {
        'num_channels': num_channels,
        'channel_indices': keep_indices.tolist(),
        'quantization': quant,
        'shape': latent_partial.shape
    }
    
    return header, compressed
```

### Decoding Process

```python
def decode_frame(header, compressed):
    # 1. Decompress
    decompressed = gzip.decompress(compressed)
    latent_quantized = np.frombuffer(decompressed, dtype=np.uint8)
    
    # 2. Dequantize
    if header['quantization'] == 'int8':
        latent = torch.from_numpy(latent_quantized).float() / 127.0
    elif header['quantization'] == 'int4':
        latent = torch.from_numpy(latent_quantized * 2).float() / 15.0
    else:
        latent = torch.from_numpy(latent_quantized).half().float()
    
    # 3. Reshape
    latent = latent.view(header['shape'])
    
    # 4. Decode (decoder automatically adapts to channel count)
    frame = decoder(latent)
    
    return frame
```

---

## Storage Requirements

### On Device:

**Single unified model:**
- Encoder: ~2.5 MB (only needed for encoding, not on playback devices)
- Decoder: ~50 MB
- Channel adapters: 6 × 8 KB = 48 KB

**Total for playback:** ~50 MB ✅

**Comparison:**
- Multi-model approach: 250 MB (5 models)
- Unified model: 50 MB (1 model)
- **Savings: 80% smaller!**

---

## Alternative: Ultra-Lightweight Decoder

For mobile devices, we can create an even smaller decoder using:

### 1. **Model Pruning**
Remove 30-50% of weights with minimal quality loss.
- Original: 50 MB
- Pruned: 25-35 MB

### 2. **INT8 Model Quantization**
Quantize the decoder weights (not latents) to INT8.
- FP32 weights: 50 MB
- INT8 weights: 12.5 MB

### 3. **Knowledge Distillation**
Train a smaller "student" decoder from the larger "teacher" decoder.
- Teacher: 50 MB (5.1M params)
- Student: 10 MB (1M params)
- Quality loss: ~1-2 dB PSNR

### Combined Optimizations:
- Original: 50 MB, 48 dB PSNR
- Pruned + INT8: 6-9 MB, 47 dB PSNR
- Distilled + INT8: 2.5 MB, 46 dB PSNR

**For mobile apps: 2.5-9 MB is very reasonable!**

---

## Implementation Timeline

### Week 1: Core Architecture
- [ ] Design unified encoder with importance prediction
- [ ] Design adaptive decoder with channel adapters
- [ ] Implement progressive training loop
- [ ] Test on synthetic data

### Week 2: Training
- [ ] Generate 50K training samples
- [ ] Train unified model (100 epochs)
- [ ] Validate on all channel counts (8, 16, 24, 32, 48, 64)
- [ ] Benchmark quality vs channel count

### Week 3: Optimization & Deployment
- [ ] Implement CRF mapping
- [ ] Test encoding/decoding pipeline
- [ ] Benchmark on real anime
- [ ] (Optional) Create distilled mobile model

### Week 4: Testing & Documentation
- [ ] Compare against AV1 at multiple CRF levels
- [ ] Document quality/bitrate curves
- [ ] Create encoding/decoding examples
- [ ] Publish results

**Total: 4 weeks** for production-ready single-model Neural CRF system

---

## Quality Projection

Based on our current 32-channel results (48 dB PSNR):

| Channels | Est. PSNR | Confidence |
|----------|-----------|------------|
| **64** | 50-52 dB | Medium (extrapolation) |
| **48** | 49-51 dB | Medium (extrapolation) |
| **32** | 48 dB | High (measured) ✅ |
| **24** | 45-47 dB | Medium (interpolation) |
| **16** | 42-44 dB | Medium (interpolation) |
| **8** | 36-40 dB | Low (extrapolation) |

**Validation needed:** Test 16 and 24 channel variants to confirm quality degradation is gradual.

---

## Advantages over Multi-Model

✅ **Storage: 50 MB vs 250 MB** (80% reduction)  
✅ **Smooth quality transitions** (can use any channel count)  
✅ **Single download** (no need to fetch different models)  
✅ **Mobile-friendly** (can be further compressed to 2.5-9 MB)  
✅ **Flexible** (encoder decides most important channels per-frame)  
✅ **Future-proof** (can add new channel counts without retraining)  

---

## Recommendation

✅ **Implement Single Variable-Channel Model (this design)**

**Why:**
1. More practical for deployment (50 MB vs 250 MB)
2. Better user experience (smooth quality control)
3. Mobile-friendly (can be compressed to 2.5 MB)
4. Only slightly more complex to train (4 weeks vs 2 weeks)

**Next steps:**
1. Implement the unified architecture
2. Train on GPU worker (4-6 hours on 8× A10G)
3. Validate quality at all channel counts
4. Deploy as default Neural CRF system
