# 🎯 LumaFlow - Next Steps

**Current Status:** 1 training video captured ✅  
**Goal:** Train LCM-based generative codec for 50-70x compression

---

## 📱 Step 1: Capture More Training Data (THIS WEEK)

### What You Need:
- **10-15 more videos** using your iPhone LumaFlow app
- **Total training data:** ~15-20 videos (you have 1, need 14 more)

### Capture Variety:

#### 🏠 Indoor Scenes (4-5 videos)
- Living room with furniture at different depths
- Kitchen with objects
- Office/desk setup
- Hallway/corridor

#### 🌳 Outdoor Scenes (4-5 videos)
- Buildings/architecture
- Trees and nature
- Street scenes
- Parks

#### 🎨 Objects & Details (4-5 videos)
- Close-up objects (phone, book, cup)
- Person sitting/standing at various distances
- Moving objects (car, person walking)
- Textured surfaces (wall, fabric, wood)

### Recording Tips:
1. **Duration:** 10-30 seconds per clip
2. **Movement:** Slow panning or stationary
3. **Lighting:** Mix of bright, dim, and mixed lighting
4. **Depth variation:** Include near/far objects in same scene

### Where Files Go:
```bash
# Your app saves to Downloads, then move them:
~/Downloads/lumaflow_*.mov → ~/lumaflow_training_data/
```

---

## 🔬 Step 2: Test Data Loading (AFTER CAPTURING 10+ VIDEOS)

### Check Python Environment:
```bash
cd /Users/yarontorbaty/Documents/Code/Aiv1-LumaFlow/generative_codec

# Install dependencies
pip install -r requirements.txt
```

### Test the Data Loader:
```bash
# This will verify your videos load correctly
python data/iphone_loader.py
```

**Expected output:**
```
✅ Loaded video: lumaflow_1761095137.mov
   - RGB frames: 359 (1920x1080)
   - Depth frames: 359 (256x192)
✅ Loaded video: lumaflow_1761095200.mov
   ...
```

---

## 🚀 Step 3: Start Training (AFTER DATA COLLECTION)

### Training Requirements:

#### **Option A: Local Training (Mac with GPU)**
- ⚠️ Only if you have Apple Silicon M1/M2/M3 Max/Ultra
- Training time: ~8-12 hours
- Free!

```bash
python train.py \
  --data_dir ~/lumaflow_training_data \
  --batch_size 2 \
  --epochs 50 \
  --device mps
```

#### **Option B: AWS GPU Instance (RECOMMENDED)**
- Instance: g4dn.xlarge ($0.526/hour)
- Training time: ~4-8 hours
- Cost: ~$4-5 total

```bash
# On AWS instance:
python train.py \
  --data_dir ~/lumaflow_training_data \
  --batch_size 4 \
  --epochs 50 \
  --device cuda
```

### Monitor Training:
```bash
# In another terminal:
tensorboard --logdir runs/lumaflow --port 6006
```

Open: http://localhost:6006

---

## 📊 Expected Training Results

### Target Metrics:

| Epoch | Train PSNR | Notes |
|-------|-----------|-------|
| 1-10 | 20-25 dB | Initial learning |
| 10-30 | 25-32 dB | Convergence |
| 30-50 | 32-38 dB | Fine-tuning |
| **Goal** | **35+ dB** | Production quality |

### Compression Performance:

- **Original video:** ~60 MB/minute (uncompressed RGB)
- **HEVC (baseline):** ~5-10 MB/minute
- **LumaFlow (target):** ~1 MB/minute (50-60x compression)

---

## 🎯 Training Timeline

### Week 1 (This Week):
- [x] Capture first video ✅
- [ ] Capture 14 more videos
- [ ] Move all videos to `~/lumaflow_training_data/`
- [ ] Test data loader

### Week 2:
- [ ] Set up training environment (AWS or local)
- [ ] Start training run
- [ ] Monitor convergence
- [ ] Checkpoint best model

### Week 3:
- [ ] Export trained model to CoreML
- [ ] Integrate into iPhone app
- [ ] Test on-device encoding
- [ ] Benchmark quality

---

## 💰 Cost Breakdown

### Development (Spent: $0)
- ✅ iPhone app built
- ✅ Codec architecture designed
- ✅ Training pipeline ready

### Training Phase (Next 2 weeks)
- AWS g4dn.xlarge: ~$4-5 (one training run)
- Storage: ~$1
- **Total: ~$5-6**

**WAY CHEAPER than the $757 estimate!** 🎉

(The $757 was for 10 weeks of continuous GPU. We only need 4-8 hours!)

---

## 🔧 Troubleshooting

### "Not enough training data"
- **Solution:** Capture at least 10 videos total
- More diversity = better generalization

### "CUDA out of memory"
- **Solution:** Reduce batch size to 2
- Or use smaller model

### "Low PSNR after 30 epochs"
- Check video quality (clear, well-lit?)
- Try lower learning rate: `--lr 5e-5`
- More training epochs: `--epochs 100`

---

## 📞 Questions?

### How many samples is ideal?
- **Minimum:** 10 videos (will work, but limited)
- **Good:** 15-20 videos (recommended)
- **Optimal:** 30+ videos (best quality)

### Can I train with just 1 video?
- Technically yes, but it will overfit
- Won't generalize to new scenes
- **Recommendation:** Capture at least 10

### How long to capture 15 videos?
- ~30 minutes of recording time
- Different locations recommended
- Can split across multiple days

---

## ✅ Current Status

**Videos Captured:** 1/15  
**Next Action:** Use your iPhone app to capture 14 more videos!  
**Time Estimate:** 30-60 minutes of capture time  
**Training ETA:** Week 2 (after you have 10+ videos)

---

## 🚀 Ready to Continue?

1. **Now:** Capture 10-15 more videos with the iPhone app
2. **After capturing:** Run data loader test
3. **After test passes:** Start training!

Your first video looks great - now let's get more training data! 📱✨
