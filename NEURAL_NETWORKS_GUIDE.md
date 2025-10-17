# 🧠 Neural Networks for Video Compression

## WHY NEURAL NETWORKS?

Traditional codecs (H.264, HEVC) use:
- **DCT** (Discrete Cosine Transform) - handcrafted math
- **Fixed quantization** - predefined tables
- **Block-based prediction** - rigid patterns

**Neural networks can learn:**
- Better transforms than DCT
- Adaptive quantization
- Semantic understanding ("this is a face, compress the background more")
- Motion prediction
- Texture synthesis (generate instead of store)

---

## AVAILABLE TOOLS (PyTorch)

The LLM can use PyTorch in its compression code:

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
```

### Example 1: Semantic Encoder

```python
class SemanticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
    
    def forward(self, x):
        # Reduce 1920x1080 frame to compact latent representation
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x  # Much smaller than original

def compress_video_frame(frame, frame_index, config):
    encoder = SemanticEncoder()
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    latent = encoder(frame_tensor.unsqueeze(0))
    compressed = latent.numpy().tobytes()  # Convert to bytes
    return compressed
```

### Example 2: Learned Transform (Replace DCT)

```python
class LearnedTransform(nn.Module):
    def __init__(self):
        super().__init__()
        # Learn a better transform than DCT
        self.analysis = nn.Conv2d(3, 192, 5, stride=2, padding=2)
        self.quantize = nn.Conv2d(192, 96, 3, stride=2, padding=1)
    
    def forward(self, x):
        coeffs = torch.tanh(self.analysis(x))
        quantized = torch.round(self.quantize(coeffs) * 255) / 255
        return quantized
```

### Example 3: Motion Prediction Network

```python
class MotionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.optical_flow = nn.Conv2d(6, 2, 7, padding=3)  # Predict motion
    
    def predict_frame(self, prev_frame, curr_frame):
        # Predict current frame from previous frame + motion
        stacked = torch.cat([prev_frame, curr_frame], dim=1)
        flow = self.optical_flow(stacked)
        # Only store flow (much smaller than full frame)
        return flow
```

### Example 4: Generative Decoder

```python
class GenerativeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Generate details instead of storing them
        self.upsample1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
    
    def forward(self, latent):
        x = torch.relu(self.upsample1(latent))
        x = torch.relu(self.upsample2(x))
        x = torch.sigmoid(self.upsample3(x))
        return x  # Reconstruct full frame from tiny latent
```

---

## COMPRESSION STRATEGIES WITH NEURAL NETWORKS

### 1. Autoencoder Compression

**Concept:** Encode frame → Tiny latent code → Decode to reconstruct

```
Original Frame (1920x1080x3 = 6.2 MB)
         ↓
    Encoder Network
         ↓
Latent Code (240x135x8 = 260 KB) ← Store this!
         ↓
    Decoder Network
         ↓
Reconstructed Frame (1920x1080x3)
```

**Compression Ratio:** ~24x (6.2 MB → 260 KB)

### 2. Motion-Compensated Prediction

**Concept:** Only store motion vectors, not full frames

```
Frame 0: Store full frame (I-frame)
Frame 1: Store motion from Frame 0 (P-frame)
Frame 2: Store motion from Frame 1 (P-frame)
...
```

**Neural network learns:**
- Better motion prediction than block matching
- Semantic motion (object-level, not block-level)

### 3. Semantic Scene Compression

**Concept:** Store scene description, not pixels

```
Frame: [Person, Background, Table]
         ↓
Semantic Segmentation Network
         ↓
Encoding:
  - Person position: 100 bytes
  - Background: "office" (10 bytes)
  - Table: 3D mesh (50 bytes)
         ↓
Decoder: Render scene from description
```

### 4. Learned Entropy Coding

**Concept:** Neural network predicts which pixels are important

```
Frame → Importance Map (neural net)
      → Adaptive quantization
      → Arithmetic coding

Important regions (faces): High quality
Unimportant (background): Low quality
```

---

## REALISTIC NEURAL NETWORK APPROACH

### Start Simple

```python
def compress_video_frame(frame, frame_index, config):
    # 1. Downscale using neural network
    downscale_net = nn.Conv2d(3, 3, 5, stride=4, padding=2)
    small = downscale_net(torch_frame)  # 1920→480, 1080→270
    
    # 2. Store small version
    compressed = small.numpy().tobytes()
    
    # Compression: 4x4 = 16x from spatial downscaling
    return compressed
```

### Intermediate

```python
def compress_video_frame(frame, frame_index, config):
    # 1. Semantic encoder
    encoder = SemanticEncoder()
    latent = encoder(frame)  # 1920x1080x3 → 240x135x64
    
    # 2. Quantize latent
    quantized = torch.round(latent * 255) / 255
    
    # 3. Compress to bytes
    compressed = quantized.numpy().astype(np.uint8).tobytes()
    
    return compressed
```

### Advanced

```python
def compress_video_frame(frame, frame_index, config):
    # 1. Motion prediction (if not first frame)
    if frame_index > 0:
        motion_net = MotionPredictor()
        residual = motion_net(prev_frame, frame)
        compressed = residual.numpy().tobytes()  # Much smaller
    else:
        # I-frame: Full compression
        encoder = SemanticEncoder()
        latent = encoder(frame)
        compressed = latent.numpy().tobytes()
    
    return compressed
```

---

## PRETRAINED MODELS

### Option 1: Use Pretrained Features

```python
from torchvision.models import resnet18

def compress_video_frame(frame, frame_index, config):
    # Use pretrained ResNet as feature extractor
    resnet = resnet18(pretrained=True)
    resnet.eval()
    
    # Extract features (much more compact than raw pixels)
    with torch.no_grad():
        features = resnet(frame_tensor)  # 1920x1080x3 → 1000 features
    
    compressed = features.numpy().tobytes()
    return compressed
```

### Option 2: Custom Learned Codec

```python
# LLM can define weights in code (small model)
class TinyCodec(nn.Module):
    def __init__(self):
        super().__init__()
        # Define small model (< 1 MB)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, 2, 1)
        )
        # Initialize with specific learned weights
        self.encoder[0].weight.data = torch.randn(16, 3, 3, 3) * 0.1
```

---

## CONSTRAINTS

### Memory
- **Available:** ~8 GB RAM on EC2 t3.large
- **Frame size:** 1920x1080x3 = 6.2 MB
- **Model size:** Keep under 100 MB (no huge pretrained models)

### Compute
- **CPU only** (no GPU on t3.large)
- **Target:** < 1 second per frame
- **Use:** Small models, efficient operations

### Storage
- **Compressed frame target:** < 400 KB (< 1 Mbps at 30fps)
- **HEVC baseline:** ~333 KB per frame at 10 Mbps

---

## LLM NEURAL NETWORK STRATEGY

### Phase 1: Simple Downscaling
- Use `nn.Conv2d` to intelligently downscale
- Better than naive resize
- 10-20x compression

### Phase 2: Semantic Encoding
- Small encoder network (3-5 layers)
- Extract compact features
- 20-40x compression

### Phase 3: Learned Transforms
- Replace DCT with learned transform
- Adaptive quantization
- 40-100x compression

### Phase 4: Motion Networks
- Predict frames from previous frames
- Store only motion/residuals
- 100-200x compression

### Phase 5: Generative Compression
- Store scene graphs
- Decoder generates pixels
- 200-500x compression (research frontier!)

---

## DEBUGGING NEURAL NETWORKS

### Common Issues

1. **Out of Memory**
   - Reduce model size
   - Process smaller patches
   - Use `torch.no_grad()` during encoding

2. **Too Slow**
   - Smaller models
   - Fewer layers
   - Use stride instead of multiple convolutions

3. **Poor Quality**
   - Start with pretrained features
   - Increase bottleneck size
   - Add skip connections

### Testing

```python
# Test on single frame first
test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

# Convert to tensor
frame_tensor = torch.from_numpy(test_frame).permute(2, 0, 1).float() / 255.0

# Test encoder
encoder = SemanticEncoder()
latent = encoder(frame_tensor.unsqueeze(0))

print(f"Original: {test_frame.nbytes / 1024 / 1024:.2f} MB")
print(f"Latent: {latent.numel() * 4 / 1024 / 1024:.2f} MB")
print(f"Compression: {test_frame.nbytes / (latent.numel() * 4):.1f}x")
```

---

## SUMMARY

**Neural Networks = Power Tools for Compression**

- ✅ Available: PyTorch 1.13.1 on EC2
- ✅ Allowed: LLM can import `torch`, `torch.nn`, `torchvision`
- ✅ Strategy: Start simple, iterate to advanced
- ✅ Goal: Beat HEVC (10 Mbps) → Approach 1 Mbps

**LLM should:**
1. Start with simple learned transforms
2. Gradually add semantic understanding
3. Eventually use motion prediction
4. Compete with state-of-art learned codecs

**Not a separate test** - Neural networks are **compression tools**! 🚀

