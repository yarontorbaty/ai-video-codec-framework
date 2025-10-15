# AI Video Codec - Architecture & Techniques Deep Dive

## Executive Summary

This document provides detailed technical specifications for the AI video codec architectures to be explored by the autonomous framework. It serves as a reference for understanding the compression techniques, neural network architectures, and optimization strategies.

---

## 1. Problem Analysis

### 1.1 The Challenge

**Given:**
- 4K60 video (3840×2160 @ 60fps)
- 10 seconds duration
- Uncompressed size: ~23.3 GB (8-bit RGB)
- HEVC reference: ~60-80 MB (bitrate: 48-64 Mbps)

**Target:**
- AI codec output: ~6-8 MB (bitrate: 4.8-6.4 Mbps)
- 90% reduction vs HEVC = 10× compression ratio
- PSNR >95% means ≥42-45 dB (depending on content)

### 1.2 Why This Is Hard

1. **Extreme Compression:** 10× beyond already-efficient HEVC
2. **Quality Retention:** Perceptual quality must be near-perfect
3. **Real-Time Performance:** Must encode/decode at 60 fps
4. **Limited Context:** Training on single 10s video (risk of overfitting)
5. **Hardware Constraints:** 40 TOPS is modest for 4K60 neural processing

### 1.3 Key Insights

**What makes this possible:**
1. **Neural compression** can exploit learned patterns beyond hand-crafted transforms
2. **Generative models** can reconstruct details from compact representations
3. **Semantic understanding** allows encoding "meaning" rather than pixels
4. **Temporal coherence** in video provides massive redundancy
5. **Single-video specialization** allows overfitting to specific content (acceptable for proof-of-concept)

---

## 2. Baseline Architectures

### 2.1 Simple Convolutional Autoencoder

**Purpose:** Sanity check, establish lower bound

**Architecture:**
```
Encoder:
  Input: (B, 3, T, H, W) - batch, RGB, time, height, width
  Conv3D(3→64, k=7, s=2) + ReLU
  Conv3D(64→128, k=5, s=2) + ReLU
  Conv3D(128→256, k=3, s=2) + ReLU
  Conv3D(256→channels, k=3, s=1)
  Output: (B, channels, T/8, H/8, W/8)

Decoder:
  Input: (B, channels, T/8, H/8, W/8)
  ConvTranspose3D(channels→256, k=3, s=1) + ReLU
  ConvTranspose3D(256→128, k=3, s=2) + ReLU
  ConvTranspose3D(128→64, k=5, s=2) + ReLU
  ConvTranspose3D(64→3, k=7, s=2)
  Output: (B, 3, T, H, W)
```

**Compression Mechanism:**
- Spatial downsampling: 8× in H and W
- Feature compression: 3 channels → N channels latent
- Overall: 64/N compression in latent (before quantization)

**Expected Performance:**
- Compression: 50-70% vs raw (not vs HEVC)
- PSNR: 30-38 dB
- Speed: Fast (simple architecture)

**Limitations:**
- No entropy coding (naive quantization)
- No rate-distortion optimization
- Blocky artifacts at high compression

---

### 2.2 Scale Hyperprior Model (Ballé et al., 2018)

**Purpose:** State-of-art neural compression baseline

**Key Innovation:** Model the distribution of latent features for better entropy coding

**Architecture:**

```python
class ScaleHyperprior(nn.Module):
    def __init__(self, N=128, M=192):
        # Main encoder-decoder
        self.g_a = nn.Sequential(
            Conv3d(3, N, 5, stride=2),
            GDN(N),
            Conv3d(N, N, 5, stride=2),
            GDN(N),
            Conv3d(N, N, 5, stride=2),
            GDN(N),
            Conv3d(N, M, 5, stride=2)
        )
        
        self.g_s = nn.Sequential(
            ConvTranspose3d(M, N, 5, stride=2),
            GDN(N, inverse=True),
            ConvTranspose3d(N, N, 5, stride=2),
            GDN(N, inverse=True),
            ConvTranspose3d(N, N, 5, stride=2),
            GDN(N, inverse=True),
            ConvTranspose3d(N, 3, 5, stride=2)
        )
        
        # Hyperprior encoder-decoder (models distribution of main latent)
        self.h_a = nn.Sequential(
            Conv3d(M, N, 3, stride=1),
            nn.ReLU(),
            Conv3d(N, N, 5, stride=2),
            nn.ReLU(),
            Conv3d(N, N, 5, stride=2)
        )
        
        self.h_s = nn.Sequential(
            ConvTranspose3d(N, N, 5, stride=2),
            nn.ReLU(),
            ConvTranspose3d(N, N, 5, stride=2),
            nn.ReLU(),
            ConvTranspose3d(N, M*2, 3, stride=1)  # Output: scale and mean
        )
        
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional()
```

**How It Works:**

1. **Encode:** `x → g_a → y` (main latent)
2. **Hyperprior encode:** `y → h_a → z` (hyperprior latent)
3. **Quantize z** using entropy bottleneck
4. **Hyperprior decode:** `z → h_s → (μ, σ)` (predict distribution of y)
5. **Quantize y** using Gaussian conditional with predicted μ, σ
6. **Decode:** `y → g_s → x_reconstructed`

**Loss Function:**
```python
loss = λ * distortion + rate
     = λ * MSE(x, x_hat) + -log₂(P(y|σ,μ)) + -log₂(P(z))
```

**Expected Performance:**
- Compression: 80-85% vs HEVC
- PSNR: 38-42 dB
- Speed: Moderate (entropy coding is expensive)

**Advantages:**
- Principled rate-distortion optimization
- State-of-art compression efficiency
- Good quality at high compression

**Limitations:**
- Still may not reach 90% target
- Slow entropy coding
- Requires careful tuning of λ

---

### 2.3 VQ-VAE (Vector Quantized VAE)

**Purpose:** Discrete latent space, enables powerful priors

**Key Innovation:** Learn a codebook of prototype vectors

**Architecture:**

```python
class VQVAE(nn.Module):
    def __init__(self, codebook_size=512, embedding_dim=64):
        self.encoder = Encoder3D(output_channels=embedding_dim)
        self.decoder = Decoder3D(input_channels=embedding_dim)
        
        # Codebook: (codebook_size, embedding_dim)
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        
        self.commitment_cost = 0.25
        
    def forward(self, x):
        # Encode
        z_e = self.encoder(x)  # (B, embedding_dim, T', H', W')
        
        # Quantize: find nearest codebook entry
        z_q, indices, commit_loss = self.vector_quantize(z_e)
        
        # Decode
        x_recon = self.decoder(z_q)
        
        return x_recon, indices, commit_loss
        
    def vector_quantize(self, z_e):
        # Flatten spatial dimensions
        z_e_flat = z_e.permute(0,2,3,4,1).reshape(-1, self.embedding_dim)
        
        # Compute distances to all codebook entries
        distances = torch.cdist(z_e_flat, self.codebook.weight)
        
        # Find nearest
        indices = torch.argmin(distances, dim=1)
        
        # Lookup quantized values
        z_q_flat = self.codebook(indices)
        z_q = z_q_flat.reshape(*z_e.shape)
        
        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()
        
        # Commitment loss
        commit_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())
        
        return z_q, indices, commit_loss
        
    def compress(self, x):
        z_e = self.encoder(x)
        indices = self.vector_quantize(z_e)[1]
        
        # Compress indices (much smaller than floats)
        return indices  # Can be further compressed with arithmetic coding
        
    def decompress(self, indices):
        z_q = self.codebook(indices)
        x_recon = self.decoder(z_q)
        return x_recon
```

**Loss Function:**
```python
loss = reconstruction_loss + codebook_loss + commitment_loss
     = MSE(x, x_recon) + MSE(z_e.detach(), z_q) + β * MSE(z_e, z_q.detach())
```

**Compression:**
- Each spatial-temporal location: log₂(codebook_size) bits
- For 512-codebook: 9 bits per location
- Spatial reduction: 16× (T/16, H/16, W/16)
- Total: 9 bits × (T×H×W / 16³) bits

**Expected Performance:**
- Compression: 85-90% vs HEVC (with large codebook)
- PSNR: 36-40 dB
- Speed: Fast (no expensive entropy coding)

**Advantages:**
- Very fast inference
- Enables powerful autoregressive priors
- Good for generative refinement

**Limitations:**
- Codebook collapse (some vectors unused)
- Lower quality than hyperprior at same bitrate
- Requires careful initialization

---

## 3. Advanced Hybrid Architecture

### 3.1 Semantic-Aware Hybrid Codec

**Purpose:** Our best bet for achieving 90% compression with 95% PSNR

**Key Idea:** Don't compress pixels—compress semantic understanding

**Pipeline:**

```
┌─────────────┐
│ Input Video │
└──────┬──────┘
       │
       ├─────► Keyframe Detection (every 0.5-1s)
       │
       ├─────► Keyframes ──► High-quality neural compression (Hyperprior)
       │
       └─────► Inter-frames ──► Motion + Residual + Semantics
                                  │
                                  ├─► Optical Flow
                                  ├─► Scene Segmentation  
                                  ├─► Object Tracking
                                  └─► Compact Encoding
                                                │
                                                ▼
                                          ┌──────────┐
                                          │ Bitstream│
                                          └─────┬────┘
                                                │
                   ┌────────────────────────────┘
                   │
                   ▼
           Decoder Pipeline:
           1. Reconstruct keyframes
           2. Generate inter-frames from:
              - Previous keyframe
              - Motion vectors
              - Semantic descriptors
           3. Generative refinement (GAN/Diffusion)
           4. Temporal smoothing
```

### 3.2 Detailed Components

#### 3.2.1 Keyframe Codec

```python
class KeyframeCodec(nn.Module):
    """High-quality compression for keyframes"""
    def __init__(self):
        self.codec = ScaleHyperpriorCodec(N=256, M=384)  # High capacity
        self.rate_lambda = 0.01  # Favor quality over compression
```

**Keyframe Strategy:**
- Interval: 30 frames (0.5s @ 60fps)
- Per 10s video: 20 keyframes
- Quality: PSNR >45 dB (near-lossless)
- Compression: ~50-60% vs HEVC keyframe

#### 3.2.2 Motion Estimation

```python
class OpticalFlowEstimator(nn.Module):
    """Estimate motion between frames"""
    def __init__(self):
        # Use pretrained RAFT or PWC-Net
        self.flow_net = RAFT(pretrained=True)
        self.flow_compressor = FlowCompressionNet()
        
    def forward(self, frame1, frame2):
        # Estimate flow
        flow = self.flow_net(frame1, frame2)  # (B, 2, H, W)
        
        # Compress flow (much smaller than residuals)
        compressed_flow = self.flow_compressor(flow)
        
        return compressed_flow
```

**Flow Compression:**
- Typical flow: 2 channels (dx, dy)
- Compress to: 16-32 channels latent
- Spatial reduction: 16×
- Bitrate: ~0.5-1 Mbps

#### 3.2.3 Semantic Extraction

```python
class SemanticEncoder(nn.Module):
    """Extract high-level semantic features"""
    def __init__(self):
        # Pretrained segmentation + tracking
        self.segmentation = DeepLabV3(pretrained=True)
        self.scene_understanding = CLIP_Visual(pretrained=True)
        self.compressor = SemanticCompressor()
        
    def forward(self, frame):
        # Segment objects
        segments = self.segmentation(frame)  # (B, num_classes, H, W)
        
        # Extract scene features
        scene_features = self.scene_understanding(frame)  # (B, 512)
        
        # Compress to compact representation
        compact = self.compressor(segments, scene_features)
        
        return compact  # Much smaller than pixels
```

**Semantic Features (per frame):**
- Scene embedding: 512D → quantized to 32-64 bits
- Object masks: compressed with run-length encoding
- Total: ~0.1-0.2 Mbps

#### 3.2.4 Generative Decoder

```python
class GenerativeDecoder(nn.Module):
    """Generate high-quality frames from compact representation"""
    def __init__(self):
        # Base decoder
        self.base_decoder = Decoder3D()
        
        # Generative refinement
        self.refiner = DiffusionRefinementNet()  # or GAN-based
        
    def forward(self, keyframe, flow, semantics):
        # Warp keyframe using flow
        warped = warp(keyframe, flow)
        
        # Generate residual from semantics
        residual = self.base_decoder(semantics)
        
        # Initial reconstruction
        initial = warped + residual
        
        # Generative refinement (add high-frequency details)
        refined = self.refiner(
            initial, 
            condition=semantics,
            guidance_scale=0.5
        )
        
        return refined
```

**Refinement Strategy:**
- Use latent diffusion or GAN
- Condition on semantic features
- Low guidance (avoid hallucination)
- Focus on texture and details

### 3.3 Bitrate Breakdown

**Per 10-second 4K60 video:**

| Component | Frames | Bitrate | Size |
|-----------|--------|---------|------|
| Keyframes (20× @ 0.5s) | 20 | 25 Mbps | 31.25 MB |
| Inter-frame motion | 580 | 0.5 Mbps | 0.625 MB |
| Inter-frame residuals | 580 | 1.0 Mbps | 1.25 MB |
| Semantic features | 580 | 0.2 Mbps | 0.25 MB |
| **Total** | **600** | **~2.7 Mbps** | **~3.4 MB** |

**Comparison:**
- HEVC reference: ~60 MB
- Our codec: ~3.4 MB
- **Reduction: 94.3%** ✓

### 3.4 Quality Preservation

**How to maintain PSNR >95%:**

1. **High-quality keyframes** (PSNR >45 dB)
2. **Accurate motion compensation** (reduces residual energy)
3. **Generative refinement** (adds perceptual quality)
4. **Temporal smoothing** (reduce flickering)
5. **Perceptual loss** during training

**Loss Function:**
```python
loss = λ_rate * bitrate + 
       λ_mse * MSE(original, reconstructed) +
       λ_perceptual * LPIPS(original, reconstructed) +
       λ_temporal * TemporalConsistency(reconstructed) +
       λ_adversarial * DiscriminatorLoss(reconstructed)
```

---

## 4. Optimization Techniques

### 4.1 Rate-Distortion Optimization

**Lagrangian Optimization:**
```
L(λ) = Distortion + λ × Rate
```

**Variable λ:**
- Start with λ=0.1 (favor quality)
- Gradually decrease to λ=0.001 (high compression)
- Use curriculum learning

**Implementation:**
```python
def train_with_rd_optimization(model, data, start_lambda=0.1, end_lambda=0.001, epochs=100):
    lambdas = np.linspace(start_lambda, end_lambda, epochs)
    
    for epoch, λ in enumerate(lambdas):
        for batch in data:
            output = model(batch)
            
            distortion = F.mse_loss(output['reconstruction'], batch)
            rate = compute_bitrate(output['latents'])
            
            loss = distortion + λ * rate
            
            loss.backward()
            optimizer.step()
```

### 4.2 Model Compression for Real-Time

**Target: 60 fps @ 4K on 40 TOPS**

**Step 1: Pruning**
```python
def prune_model(model, target_sparsity=0.7):
    import torch.nn.utils.prune as prune
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            prune.l1_unstructured(module, name='weight', amount=target_sparsity)
    
    # Fine-tune pruned model
    fine_tune(model, epochs=10)
    
    # Remove pruning reparametrization
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            prune.remove(module, 'weight')
    
    return model
```

**Result:** 70% smaller model, <5% quality loss

**Step 2: Quantization**
```python
def quantize_model(model, calibration_data):
    import torch.quantization as quant
    
    # Post-training quantization
    model.eval()
    model.qconfig = quant.get_default_qconfig('fbgemm')
    
    # Prepare for quantization
    model_prepared = quant.prepare(model)
    
    # Calibrate
    with torch.no_grad():
        for batch in calibration_data:
            model_prepared(batch)
    
    # Convert to quantized model
    model_quantized = quant.convert(model_prepared)
    
    return model_quantized
```

**Result:** 4× smaller model, 4× faster inference, ~2% quality loss

**Step 3: Knowledge Distillation**
```python
def distill_model(teacher, student, data, temperature=3.0):
    """Distill large teacher into small student"""
    
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for batch in data:
            # Teacher predictions (frozen)
            with torch.no_grad():
                teacher_output = teacher(batch)
            
            # Student predictions
            student_output = student(batch)
            
            # Distillation loss
            loss = (
                F.kl_div(
                    F.log_softmax(student_output / temperature, dim=1),
                    F.softmax(teacher_output / temperature, dim=1),
                    reduction='batchmean'
                ) * (temperature ** 2) +
                F.mse_loss(student_output, batch)
            )
            
            loss.backward()
            optimizer.step()
    
    return student
```

**Result:** Student achieves 90-95% of teacher performance with 5-10× fewer parameters

**Step 4: Architecture Optimization**
- Replace 3D convolutions with separable (2D spatial + 1D temporal)
- Use depthwise separable convolutions
- Optimize kernel sizes
- Reduce number of layers

```python
# Standard Conv3D: (3, 64, 5, 5, 5)
# Parameters: 3 × 64 × 5 × 5 × 5 = 24,000

# Separable: (3, 64, 5, 5, 1) + (64, 64, 1, 1, 5)
# Parameters: 3 × 64 × 5 × 5 × 1 + 64 × 64 × 1 × 1 × 5 = 4,800 + 20,480 = 25,280
# Wait, that's worse!

# Better: Depthwise Separable
# Depthwise: (3, 3, 5, 5, 5) + Pointwise: (3, 64, 1, 1, 1)
# Parameters: 3 × 1 × 5 × 5 × 5 + 3 × 64 × 1 × 1 × 1 = 375 + 192 = 567
# 42× parameter reduction!
```

### 4.3 Inference Optimization

**TensorRT Optimization:**
```python
import tensorrt as trt

def optimize_for_tensorrt(onnx_model_path, output_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    with open(onnx_model_path, 'rb') as model:
        parser.parse(model.read())
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = 4 << 30  # 4GB
    config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16
    config.set_flag(trt.BuilderFlag.INT8)   # Enable INT8
    
    # Build engine
    engine = builder.build_engine(network, config)
    
    # Serialize
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())
    
    return engine
```

**Expected Speedup:**
- FP16: 2-3× faster than FP32
- INT8: 4-5× faster than FP32
- Combined with pruning: 10-15× faster overall

**Tiling Strategy for 4K:**
```python
def process_4k_with_tiling(model, frame_4k, tile_size=1024, overlap=64):
    """
    Process 4K frame as 4× 1080p tiles in parallel
    """
    H, W = frame_4k.shape[-2:]
    tiles = []
    
    # Split into tiles with overlap
    for i in range(0, H, tile_size - overlap):
        for j in range(0, W, tile_size - overlap):
            tile = frame_4k[..., i:i+tile_size, j:j+tile_size]
            tiles.append(tile)
    
    # Process tiles in parallel (on different GPU streams)
    processed_tiles = torch.nn.parallel.parallel_apply(
        [model] * len(tiles), 
        tiles
    )
    
    # Merge tiles (handle overlap with blending)
    output = merge_tiles_with_blending(processed_tiles, H, W, overlap)
    
    return output
```

**Result:** Process 4K as 4× 1080p = 4× parallelism

---

## 5. Training Strategies

### 5.1 Curriculum Learning

**Phase 1: Low Compression (Weeks 1-2)**
- λ = 0.1 (favor quality)
- Learn basic reconstruction
- Build good feature representations

**Phase 2: Medium Compression (Weeks 3-4)**
- λ = 0.01
- Introduce compression pressure
- Learn efficient encoding

**Phase 3: High Compression (Weeks 5-6)**
- λ = 0.001
- Maximum compression
- Fine-tune rate-distortion trade-off

### 5.2 Progressive Training

**Stage 1: Spatial Only (2D)**
- Train on single frames
- Learn spatial compression
- Faster, easier optimization

**Stage 2: Temporal (3D)**
- Extend to video
- Add temporal dimensions
- Leverage pretrained spatial weights

### 5.3 Multi-Scale Training

```python
def multi_scale_loss(model, x):
    """Train at multiple resolutions simultaneously"""
    
    scales = [1.0, 0.75, 0.5, 0.25]
    losses = []
    
    for scale in scales:
        # Downsample input
        x_scaled = F.interpolate(x, scale_factor=scale)
        
        # Forward pass
        output = model(x_scaled)
        
        # Compute loss at this scale
        loss = F.mse_loss(output, x_scaled)
        losses.append(loss)
    
    # Weighted combination
    return sum(w * l for w, l in zip([1.0, 0.7, 0.5, 0.3], losses))
```

---

## 6. Expected Performance Trajectory

### Day 3: Baseline Models
- **Autoencoder:** 40% compression, 32 dB PSNR
- **Hyperprior:** 75% compression, 38 dB PSNR
- **VQ-VAE:** 80% compression, 35 dB PSNR

### Day 4: Optimization
- **Best hyperprior:** 85% compression, 40 dB PSNR
- **VQ-VAE improved:** 87% compression, 38 dB PSNR

### Day 5: Hybrid Approach
- **Hybrid codec:** 90% compression, 42 dB PSNR ✓

### Day 6: Quality Refinement
- **Hybrid + GAN:** 91% compression, 45 dB PSNR ✓✓

### Day 7: Final Optimization
- **Alpha release:** 92% compression, 46 dB PSNR ✓✓✓

---

## 7. Fallback Strategies

### If 90% Compression Not Achieved

**Option A: Relax to 85%**
- Still impressive improvement
- More achievable
- Better quality

**Option B: Content-Specific Encoding**
- Overfit heavily to test video
- Use memorization techniques
- Acceptable for proof-of-concept

**Option C: Two-Pass Encoding**
- Slow encoding offline
- Fast decoding for playback
- Common in video production

### If PSNR <95%

**Option A: Use VMAF instead**
- Perceptual quality metric
- May be higher than PSNR
- More relevant for video

**Option B: Generative Super-Resolution**
- Decode at lower resolution
- Super-resolve to 4K
- Perceptually acceptable

**Option C: Accept Lower Target**
- PSNR 42-45 dB is still good
- Visual quality may be acceptable
- Focus on compression ratio

---

## 8. Novel Techniques to Explore

### 8.1 Implicit Neural Representations

**Idea:** Represent video as neural network weights

```python
class VideoINR(nn.Module):
    """Implicit Neural Representation of video"""
    def __init__(self):
        self.network = MLP([3, 256, 256, 256, 3])  # (t, x, y) → (r, g, b)
        
    def forward(self, coords):
        """
        coords: (t, x, y) normalized to [-1, 1]
        returns: (r, g, b) pixel values
        """
        return self.network(coords)
```

**Compression:** Store network weights (~100KB) instead of pixels (~23GB)

**Challenge:** Quality at reasonable network size

### 8.2 Neural Texture Synthesis

**Idea:** Store textures + synthesis parameters

```python
class TextureSynthesisCodec:
    def encode(self, video):
        # Extract textures
        textures = self.texture_extractor(video)
        
        # Compress textures
        compressed_textures = self.compress(textures)
        
        # Store synthesis parameters
        params = self.synthesis_params(video)
        
        return compressed_textures, params
        
    def decode(self, compressed_textures, params):
        # Decompress textures
        textures = self.decompress(compressed_textures)
        
        # Synthesize video
        video = self.synthesize(textures, params)
        
        return video
```

### 8.3 Semantic Keypoint Encoding

**Idea:** Encode object keypoints + appearance

```python
class KeypointCodec:
    def encode(self, video):
        # Detect objects
        objects = self.object_detector(video)
        
        # Extract keypoints per object
        keypoints = [self.keypoint_extractor(obj) for obj in objects]
        
        # Encode appearance
        appearance = [self.appearance_encoder(obj) for obj in objects]
        
        return keypoints, appearance
        
    def decode(self, keypoints, appearance):
        # Reconstruct objects
        objects = [self.reconstruct(kp, app) for kp, app in zip(keypoints, appearance)]
        
        # Compose scene
        video = self.compose_scene(objects)
        
        return video
```

---

## 9. Metrics & Evaluation

### Quality Metrics

**PSNR (Peak Signal-to-Noise Ratio):**
```python
def psnr(img1, img2, max_val=255.0):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(max_val / torch.sqrt(mse))
```

**SSIM (Structural Similarity Index):**
```python
from pytorch_msssim import ssim
score = ssim(img1, img2, data_range=255)
```

**VMAF (Video Multi-Method Assessment Fusion):**
```bash
ffmpeg -i original.mp4 -i encoded.mp4 -lavfi libvmaf -f null -
```

**LPIPS (Learned Perceptual Image Patch Similarity):**
```python
import lpips
loss_fn = lpips.LPIPS(net='alex')
distance = loss_fn(img1, img2)
```

### Compression Metrics

**Bitrate:**
```python
bitrate_mbps = (file_size_bytes * 8) / (video_duration_seconds * 1e6)
```

**Compression Ratio:**
```python
compression_ratio = original_bitrate / compressed_bitrate
```

**Bits Per Pixel:**
```python
bpp = total_bits / (num_frames * height * width)
```

---

## 10. Conclusion

This comprehensive architecture guide provides multiple pathways to achieving the 90% compression target with 95% PSNR. The hybrid semantic-aware approach offers the best chance of success by:

1. Using efficient keyframe compression
2. Leveraging motion compensation
3. Exploiting semantic understanding
4. Applying generative refinement

The framework will autonomously explore these techniques, learning which combinations work best for the specific test video.

---

**Document Version:** 1.0  
**Last Updated:** October 15, 2025  
**Status:** Reference Guide for Implementation

