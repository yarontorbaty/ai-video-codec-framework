# Neural Rate Control: CRF vs Bitrate Modes

## Overview

Traditional codecs offer two encoding modes:
1. **CRF Mode (Constant Rate Factor):** Target a quality level, bitrate varies
2. **Bitrate Mode (CBR/VBR):** Target a bitrate, quality varies

Let's implement both for our neural codec!

---

## Mode 1: CRF Mode (Quality-Based)

**User specifies:** Target quality (Neural CRF 0-51)  
**Codec decides:** How many channels/bits to use  
**Result:** Consistent quality, variable bitrate

### Usage:
```bash
pvc_encode input.mp4 output.pvc --crf 18
# High quality, ~12 Mbps for 1080p @ 30fps
```

**Implementation:** (Already designed in NEURAL_CRF_SINGLE_MODEL.md)
- CRF 18 → 32 channels → ~12.6 KB per frame
- Quality guaranteed, bitrate varies by scene complexity

---

## Mode 2: Bitrate Mode (Size-Based)

**User specifies:** Target bitrate (e.g., 5 Mbps, 10 Mbps)  
**Codec decides:** How many channels to use per frame  
**Result:** Consistent bitrate, variable quality

### Two Approaches:

### **A) Two-Pass Encoding (Better quality)**
Pass 1: Analyze entire video  
Pass 2: Encode with optimal channel allocation

### **B) Single-Pass Encoding (Faster)**
Real-time adaptive channel allocation

---

## Two-Pass Bitrate Mode (Recommended)

### Pass 1: Analysis & Budget Allocation

```python
def analyze_video(video_path, target_bitrate_mbps, fps=30):
    """
    Analyze video to determine complexity per frame and allocate bitrate budget.
    
    Args:
        video_path: Input video
        target_bitrate_mbps: Target bitrate in Mbps (e.g., 5, 10, 15)
        fps: Frames per second
    
    Returns:
        frame_budgets: List of byte budgets per frame
    """
    # 1. Calculate total bitrate budget
    target_bytes_per_frame = (target_bitrate_mbps * 1_000_000) / (8 * fps)
    # Example: 10 Mbps @ 30fps = 41,666 bytes/frame
    
    # 2. Extract all frames and measure complexity
    frames = load_video(video_path)
    complexities = []
    
    for frame in frames:
        # Measure complexity (multiple methods)
        # A) Temporal difference (motion)
        temporal_diff = calculate_temporal_diff(frame, prev_frame)
        
        # B) Spatial complexity (detail)
        spatial_complexity = calculate_spatial_complexity(frame)
        
        # C) Neural prediction (encoder's perspective)
        with torch.no_grad():
            latent, _, importance = encoder(frame)
            # Higher variance in importance = more complex
            neural_complexity = importance.std().item()
        
        # Combined complexity score
        complexity = (0.3 * temporal_diff + 
                     0.3 * spatial_complexity + 
                     0.4 * neural_complexity)
        complexities.append(complexity)
    
    # 3. Allocate budget proportional to complexity
    total_complexity = sum(complexities)
    frame_budgets = []
    
    for complexity in complexities:
        # More complex frames get more bits
        budget = target_bytes_per_frame * (complexity / (total_complexity / len(frames)))
        
        # Constrain to reasonable range (min 4 KB, max 50 KB for 960x540)
        budget = max(4_000, min(50_000, budget))
        
        frame_budgets.append(budget)
    
    # 4. Normalize to hit exact target bitrate
    actual_avg = sum(frame_budgets) / len(frame_budgets)
    adjustment = target_bytes_per_frame / actual_avg
    frame_budgets = [b * adjustment for b in frame_budgets]
    
    return frame_budgets

def calculate_temporal_diff(frame, prev_frame):
    """Measure motion/change from previous frame."""
    if prev_frame is None:
        return 1.0
    diff = np.abs(frame.astype(float) - prev_frame.astype(float)).mean()
    return diff / 255.0  # Normalize

def calculate_spatial_complexity(frame):
    """Measure detail/texture in frame."""
    # Use Laplacian variance (edge detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return min(variance / 1000.0, 2.0)  # Normalize, cap at 2.0
```

### Pass 2: Adaptive Encoding

```python
def encode_with_budget(frame, budget_bytes):
    """
    Encode frame to fit within byte budget by adjusting channel count.
    
    Args:
        frame: Input frame (960x540)
        budget_bytes: Target size in bytes (e.g., 8000 = 8 KB)
    
    Returns:
        compressed: Encoded frame data
        actual_size: Actual compressed size
        channels_used: Number of channels used
    """
    # 1. Encode to full latent
    latent_full, sorted_indices, importance = encoder(frame)
    
    # 2. Binary search for optimal channel count
    channel_counts = [8, 16, 24, 32, 48, 64]
    
    # Quick estimate: bytes ≈ (H × W × channels × 0.5) / 4
    # For 960×540: bytes ≈ (30 × 17 × channels × 0.5) / 4 = 63.75 × channels
    estimated_channels = int(budget_bytes / 63.75)
    
    # Find closest available channel count
    best_ch = min(channel_counts, key=lambda x: abs(x - estimated_channels))
    
    # Try this channel count and neighbors
    candidates = []
    for ch in [best_ch // 2, best_ch, best_ch * 2]:
        if ch not in channel_counts:
            continue
        
        # Encode with this channel count
        keep_indices = sorted_indices[:ch]
        latent_partial = latent_full[:, keep_indices, :, :]
        
        # Quantize and compress
        latent_int8 = (latent_partial * 127).byte()
        compressed = gzip.compress(latent_int8.cpu().numpy().tobytes())
        
        size = len(compressed)
        candidates.append((ch, compressed, size))
    
    # 3. Select the one closest to budget (prefer slightly under)
    best_candidate = min(candidates, 
                        key=lambda x: abs(x[2] - budget_bytes) if x[2] <= budget_bytes * 1.1
                        else abs(x[2] - budget_bytes) + 100000)
    
    channels_used, compressed, actual_size = best_candidate
    
    return compressed, actual_size, channels_used
```

### Complete Two-Pass Pipeline

```python
def encode_video_bitrate_mode(input_video, output_file, target_mbps, fps=30):
    """
    Two-pass encoding with target bitrate.
    
    Args:
        input_video: Path to input video
        output_file: Path to output .pvc file
        target_mbps: Target bitrate in Mbps (e.g., 5, 10, 15)
        fps: Frames per second
    """
    print(f"=== PASS 1: Analyzing video complexity ===")
    frame_budgets = analyze_video(input_video, target_mbps, fps)
    
    print(f"=== PASS 2: Encoding with adaptive quality ===")
    frames = load_video(input_video)
    compressed_frames = []
    
    for i, (frame, budget) in enumerate(zip(frames, frame_budgets)):
        compressed, actual_size, channels = encode_with_budget(frame, budget)
        compressed_frames.append({
            'data': compressed,
            'channels': channels,
            'size': actual_size,
            'budget': budget
        })
        
        if i % 100 == 0:
            avg_size = sum(f['size'] for f in compressed_frames) / len(compressed_frames)
            current_mbps = (avg_size * 8 * fps) / 1_000_000
            print(f"Frame {i}/{len(frames)}: "
                  f"{channels} channels, "
                  f"{actual_size} bytes, "
                  f"avg bitrate: {current_mbps:.2f} Mbps")
    
    # Write to file
    save_pvc_file(output_file, compressed_frames, metadata={
        'fps': fps,
        'target_bitrate': target_mbps,
        'mode': 'two_pass_vbr'
    })
    
    # Report final bitrate
    avg_size = sum(f['size'] for f in compressed_frames) / len(compressed_frames)
    actual_mbps = (avg_size * 8 * fps) / 1_000_000
    print(f"\n✅ Encoding complete!")
    print(f"Target: {target_mbps} Mbps, Actual: {actual_mbps:.2f} Mbps")
```

### Usage Examples:

```bash
# Two-pass VBR encoding
pvc_encode input.mp4 output.pvc --bitrate 10M --passes 2

# Target 10 Mbps for 1080p @ 30fps
pvc_encode input.mp4 output.pvc -b 10M

# Lower bitrate for streaming
pvc_encode input.mp4 output.pvc -b 5M --preset fast
```

---

## Single-Pass Bitrate Mode

For live encoding or faster processing:

```python
def encode_single_pass_cbr(frame, target_bytes, rate_buffer):
    """
    Single-pass encoding with rate control buffer.
    
    Args:
        frame: Current frame
        target_bytes: Target bytes per frame
        rate_buffer: Accumulated byte deficit/surplus
    
    Returns:
        compressed, actual_size, new_buffer
    """
    # Adjust budget based on buffer state
    # If we're over budget (buffer < 0), use fewer channels
    # If we're under budget (buffer > 0), use more channels
    
    adjusted_budget = target_bytes + (rate_buffer * 0.5)
    adjusted_budget = max(4_000, min(50_000, adjusted_budget))
    
    # Encode with adjusted budget
    compressed, actual_size, channels = encode_with_budget(frame, adjusted_budget)
    
    # Update buffer
    new_buffer = rate_buffer + (target_bytes - actual_size)
    
    # Prevent buffer from growing too large
    new_buffer = max(-target_bytes * 5, min(target_bytes * 5, new_buffer))
    
    return compressed, actual_size, channels, new_buffer
```

**Usage:**
```bash
# Single-pass CBR (faster, slightly lower quality)
pvc_encode input.mp4 output.pvc --bitrate 10M --passes 1
```

---

## Unified Command-Line Interface

```bash
# CRF Mode (quality-based, variable bitrate)
pvc_encode input.mp4 output.pvc --crf 18
pvc_encode input.mp4 output.pvc --crf 23  # Lower quality, smaller file

# Bitrate Mode (size-based, variable quality)
pvc_encode input.mp4 output.pvc --bitrate 10M
pvc_encode input.mp4 output.pvc -b 5M     # Short form

# Two-pass for best quality at target bitrate
pvc_encode input.mp4 output.pvc -b 10M --passes 2

# Constrained quality (hybrid: target bitrate but with quality floor)
pvc_encode input.mp4 output.pvc -b 10M --crf-max 28

# Auto mode: analyze and recommend
pvc_encode input.mp4 output.pvc --auto
# Analyzes video, suggests: "Recommended: --crf 20 (avg 8.5 Mbps) or -b 10M"
```

---

## Bitrate Ladder (Like Streaming Services)

Generate multiple quality levels automatically:

```python
def create_bitrate_ladder(input_video, output_dir):
    """
    Generate multiple quality levels for adaptive streaming.
    """
    profiles = [
        # Resolution, Bitrate, Name
        ('360p',  1.5, 'low'),
        ('480p',  3.0, 'medium'),
        ('720p',  6.0, 'high'),
        ('1080p', 10.0, 'ultra'),
    ]
    
    for resolution, bitrate_mbps, name in profiles:
        output_file = f"{output_dir}/{name}.pvc"
        
        # Resize + encode
        resized_video = resize_video(input_video, resolution)
        encode_video_bitrate_mode(resized_video, output_file, bitrate_mbps)
        
        print(f"✅ {name}: {resolution} @ {bitrate_mbps} Mbps")
```

**Usage:**
```bash
pvc_encode input.mp4 output_dir/ --ladder
# Creates: low.pvc, medium.pvc, high.pvc, ultra.pvc
```

---

## Rate-Distortion Optimization

For advanced users, directly optimize the rate-distortion tradeoff:

```python
def rate_distortion_optimize(frame, lambda_rd=0.01):
    """
    Optimize rate-distortion tradeoff using Lagrangian multiplier.
    
    RD Cost = Distortion + λ × Rate
    
    Args:
        frame: Input frame
        lambda_rd: Lagrangian multiplier (higher = prefer smaller files)
    
    Returns:
        Best channel count that minimizes RD cost
    """
    latent_full, sorted_indices, _ = encoder(frame)
    
    best_cost = float('inf')
    best_channels = 32
    
    for num_channels in [8, 16, 24, 32, 48, 64]:
        # Encode with this channel count
        keep_indices = sorted_indices[:num_channels]
        latent_partial = latent_full[:, keep_indices, :, :]
        
        # Decode
        reconstructed = decoder(latent_partial)
        
        # Measure distortion (MSE)
        mse = torch.mean((frame - reconstructed) ** 2).item()
        
        # Estimate rate (bytes)
        latent_int8 = (latent_partial * 127).byte()
        compressed = gzip.compress(latent_int8.cpu().numpy().tobytes())
        rate = len(compressed)
        
        # RD Cost
        rd_cost = mse + lambda_rd * rate
        
        if rd_cost < best_cost:
            best_cost = rd_cost
            best_channels = num_channels
    
    return best_channels
```

**Usage:**
```bash
# Optimize for quality (small λ)
pvc_encode input.mp4 output.pvc --rd-lambda 0.001

# Optimize for size (large λ)
pvc_encode input.mp4 output.pvc --rd-lambda 0.1
```

---

## Comparison: CRF vs Bitrate Mode

| Mode | User Specifies | Codec Controls | Use Case |
|------|---------------|----------------|----------|
| **CRF** | Quality (0-51) | Bitrate varies | Archival, high-quality encodes |
| **Bitrate (VBR)** | Target bitrate | Quality varies per frame | Streaming, file size limits |
| **Bitrate (CBR)** | Fixed bitrate | Quality varies (strict) | Live streaming, bandwidth limits |
| **Constrained CRF** | Quality + max bitrate | Hybrid | Broadcasting, premium streaming |

### When to Use Each:

**CRF Mode:**
- ✅ Archival/preservation (CRF 15-18)
- ✅ High-quality downloads (CRF 18-23)
- ✅ Don't care about file size
- ✅ Want consistent quality

**Bitrate Mode:**
- ✅ Streaming (10 Mbps for 1080p)
- ✅ File size constraints (fit on device)
- ✅ Bandwidth limits (5 Mbps mobile)
- ✅ Adaptive bitrate streaming (ladder)

---

## Implementation Priority

### Phase 1 (Week 1-2): CRF Mode
- [x] Design single-model architecture
- [x] CRF-to-channel mapping
- [ ] Train unified model
- [ ] Validate quality at all channel counts

### Phase 2 (Week 3): Bitrate Analysis
- [ ] Implement complexity analysis
- [ ] Implement two-pass bitrate mode
- [ ] Test on real videos
- [ ] Benchmark bitrate accuracy

### Phase 3 (Week 4): Single-Pass Mode
- [ ] Implement rate buffer
- [ ] Single-pass CBR encoding
- [ ] Test on live streams
- [ ] Compare vs two-pass

### Phase 4 (Week 5): Advanced Features
- [ ] Constrained quality mode
- [ ] Bitrate ladder generation
- [ ] RD optimization
- [ ] Adaptive streaming support

---

## Unified Encoder API

```python
class PVCEncoder:
    def __init__(self, model_path):
        self.encoder = load_encoder(model_path)
        self.decoder = load_decoder(model_path)
    
    def encode(self, input_video, output_file, **kwargs):
        """
        Unified encoding API supporting multiple modes.
        
        Modes:
            crf: Quality-based (--crf 18)
            bitrate: Size-based (--bitrate 10M)
            rd: Rate-distortion optimized (--rd-lambda 0.01)
            auto: Automatic mode selection
        """
        mode = kwargs.get('mode', 'crf')
        
        if mode == 'crf':
            return self._encode_crf(input_video, output_file, kwargs['crf'])
        
        elif mode == 'bitrate':
            passes = kwargs.get('passes', 2)
            if passes == 2:
                return self._encode_bitrate_twopass(
                    input_video, output_file, kwargs['bitrate'])
            else:
                return self._encode_bitrate_singlepass(
                    input_video, output_file, kwargs['bitrate'])
        
        elif mode == 'rd':
            return self._encode_rd_optimize(
                input_video, output_file, kwargs['lambda'])
        
        elif mode == 'auto':
            return self._encode_auto(input_video, output_file)

# Usage examples
encoder = PVCEncoder('model.pth')

# CRF mode
encoder.encode('input.mp4', 'out.pvc', mode='crf', crf=18)

# Bitrate mode (two-pass)
encoder.encode('input.mp4', 'out.pvc', mode='bitrate', bitrate='10M', passes=2)

# RD optimization
encoder.encode('input.mp4', 'out.pvc', mode='rd', lambda_rd=0.01)

# Auto mode
encoder.encode('input.mp4', 'out.pvc', mode='auto')
```

---

## Conclusion

✅ **CRF Mode:** Quality-first, simple, good for archival  
✅ **Bitrate Mode (2-pass):** Best quality at target bitrate  
✅ **Bitrate Mode (1-pass):** Faster, good for live streaming  
✅ **Hybrid Modes:** Constrained quality, RD optimization  

**Recommendation:** Implement both modes, defaulting to CRF 18 for simplicity, with bitrate mode for streaming use cases.

**Timeline:**
- CRF mode: 2 weeks (already designed)
- + Bitrate mode: +2 weeks
- + Advanced features: +1 week
- **Total: 5 weeks** for full rate control system
