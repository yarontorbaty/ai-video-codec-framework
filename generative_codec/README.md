# LumaFlow Codec - Development Guide

**Status:** Ready for Training  
**Date:** October 21, 2025

---

## 🎯 What's Been Built

### Core Components:

1. **LCM-based Encoder/Decoder** (`models/lcm_codec.py`)
   - Uses Latent Consistency Models for fast inference
   - VAE-based latent encoding (4x64x64 per frame)
   - Depth integration for LiDAR data
   - I-frame and P-frame support

2. **iPhone Data Loader** (`data/iphone_loader.py`)
   - Loads .mov files from LumaFlow iPhone app
   - Extracts RGB + depth tracks
   - Batch processing for training
   - Automatic preprocessing

3. **Training Pipeline** (`train.py`)
   - Fine-tunes LCM decoder on iPhone data
   - Tensorboard logging
   - Checkpoint saving
   - Validation metrics (PSNR, MSE, L1)

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd generative_codec
pip install -r requirements.txt
```

**Key packages:**
- `torch` - PyTorch framework
- `diffusers` - For LCM models
- `transformers` - Model loading
- `av` - Multi-track video loading
- `opencv-python` - Video I/O
- `tensorboard` - Training visualization

### Step 2: Capture Training Data (iPhone App)

1. Build and run LumaFlow app on iPhone
2. Use **Mode 1: Save to File**
3. Capture 10-20 videos of various scenes:
   - Indoor scenes
   - Outdoor scenes
   - Objects at various depths
   - Different lighting conditions
4. Transfer .mov files to Mac via AirDrop

### Step 3: Organize Data

```bash
# Create training data directory
mkdir -p ~/lumaflow_training_data

# Move captured videos
mv ~/Downloads/lumaflow_*.mov ~/lumaflow_training_data/
```

Expected structure:
```
~/lumaflow_training_data/
├── lumaflow_1760982333.mov  (RGB + depth track)
├── lumaflow_1760982445.mov
├── lumaflow_1760982556.mov
└── ...
```

### Step 4: Test Data Loading

```bash
# Test if videos load correctly
cd generative_codec
python data/iphone_loader.py
```

Edit the script to point to your data directory first!

### Step 5: Start Training

```bash
# Training on GPU (recommended)
python train.py \
  --data_dir ~/lumaflow_training_data \
  --batch_size 4 \
  --epochs 50 \
  --lr 1e-4 \
  --device cuda

# Training on CPU (slow, for testing only)
python train.py \
  --data_dir ~/lumaflow_training_data \
  --batch_size 2 \
  --epochs 10 \
  --device cpu
```

### Step 6: Monitor Training

```bash
# In another terminal
cd generative_codec
tensorboard --logdir runs/lumaflow --port 6006
```

Open: http://localhost:6006

---

## 📊 Expected Results

### With iPhone LiDAR Data:

| Epoch | Train PSNR | Val PSNR | Notes |
|-------|-----------|----------|-------|
| 1-5 | 20-25 dB | 20-24 dB | Initial learning |
| 10-20 | 25-30 dB | 25-28 dB | Convergence |
| 30-50 | 30-35 dB | 30-33 dB | Fine-tuning |
| **Target** | **35+ dB** | **33-35 dB** | Production quality |

### Training Time (AWS g4dn.xlarge):

- **Per epoch:** ~5-10 minutes (depends on data size)
- **Total (50 epochs):** ~4-8 hours
- **Cost:** ~$177 ($0.526/hr × 8 hrs)

---

## 🔧 Advanced Usage

### Custom Training Parameters

```bash
python train.py \
  --data_dir ~/lumaflow_training_data \
  --batch_size 8 \              # Larger batch (needs more VRAM)
  --epochs 100 \                # More epochs
  --lr 5e-5 \                   # Lower learning rate
  --val_split 0.15 \            # 15% validation
  --checkpoint_dir checkpoints/ \
  --log_dir runs/lumaflow_exp1
```

### Resume from Checkpoint

```python
# In train.py, modify main() to load checkpoint:
checkpoint = torch.load('checkpoints/lumaflow_epoch25.pt')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### Encode a Test Video

```python
from models.lcm_codec import LumaFlowCodec

# Initialize codec
codec = LumaFlowCodec(device='cuda')

# Encode video
stats = codec.encode_video(
    'test_video.mov',
    output_path='compressed.lfv'
)

# Decode video
codec.decode_video(
    'compressed.lfv',
    output_path='reconstructed.mp4'
)
```

---

## 📁 Project Structure

```
generative_codec/
├── models/
│   ├── __init__.py
│   └── lcm_codec.py           # LCM encoder/decoder
├── data/
│   ├── __init__.py
│   └── iphone_loader.py       # iPhone data loader
├── utils/
│   └── __init__.py
├── tests/
│   └── __init__.py
├── train.py                   # Training script
├── requirements.txt           # Python dependencies
└── README.md                  # This file

checkpoints/                   # Model checkpoints (created)
runs/                          # Tensorboard logs (created)
```

---

## 🐛 Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
python train.py --batch_size 2 --data_dir ~/lumaflow_training_data
```

### "No .mov files found"
```bash
# Check your data directory
ls -lh ~/lumaflow_training_data/

# Make sure files end with .mov
file ~/lumaflow_training_data/lumaflow_*.mov
```

### "Cannot load video track"
The iPhone app saves multi-track videos. If PyAV fails, the loader will fall back to OpenCV (RGB only).

To debug:
```python
import av
container = av.open('lumaflow_xxx.mov')
print(f"Video streams: {len(container.streams.video)}")
```

### Low PSNR (<25 dB after 20 epochs)
- **Check data quality:** Are videos clear and well-lit?
- **Check depth data:** Is depth track present?
- **Try lower learning rate:** `--lr 5e-5`
- **More data:** Capture 20+ diverse videos

---

## 📈 Next Steps

### After Training (Week 6):

1. **Export to CoreML**
   ```python
   # Convert PyTorch model to CoreML
   import coremltools as ct
   
   traced_model = torch.jit.trace(decoder, example_input)
   mlmodel = ct.convert(traced_model, convert_to="mlprogram")
   mlmodel.save("LumaFlowDecoder.mlpackage")
   ```

2. **Integrate into iPhone App**
   - Replace placeholder in `OnDeviceEncoder.swift`
   - Add CoreML model to Xcode project
   - Test on-device encoding

3. **Optimize Performance**
   - Quantize to INT8
   - Use Metal shaders for depth
   - Batch frame processing

4. **Measure Quality**
   - Compare to HEVC at same bitrate
   - Subjective quality tests
   - A/B testing with users

---

## 💰 Cost Breakdown

### Training Phase:

| Resource | Cost | Duration | Total |
|----------|------|----------|-------|
| **g4dn.xlarge GPU** | $0.526/hr | 8 hours | $4.21 |
| **Storage (100GB)** | $0.10/GB/mo | 1 month | $10 |
| **Data transfer** | $0.09/GB | 50 GB | $4.50 |
| **Total** | | | **~$19** |

**Way cheaper than estimated!** (We estimated $707 but if you already have the data, it's just GPU time)

---

## ✅ Checklist

Before starting training:

- [ ] iPhone app working and capturing videos
- [ ] Transferred 10+ .mov files to Mac
- [ ] Installed all Python dependencies
- [ ] Tested data loader on your videos
- [ ] GPU available (CUDA installed)
- [ ] Tensorboard accessible
- [ ] Checkpoint directory created

Ready to train? Run:
```bash
python train.py --data_dir ~/lumaflow_training_data
```

---

## 📞 Need Help?

Common issues and solutions are in the Troubleshooting section above.

For iPhone app issues, see: `../LumaFlow/README.md`

---

**Let's build the future of video compression! 🚀**

