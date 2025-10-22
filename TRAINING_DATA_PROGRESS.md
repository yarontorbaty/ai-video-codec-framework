# 📊 LumaFlow Training Data - Progress Tracker

**Last Updated:** October 21, 2025

---

## ✅ Current Status

### Videos Captured: **2 / 15** (13% complete)

| # | Filename | Size | Duration | Scene Type | Depth Quality |
|---|----------|------|----------|------------|---------------|
| 1 | `lumaflow_1761095137.mov` | 19 MB | ~13s | Unknown | ⚠️ Green artifacts (old) |
| 2 | `lumaflow_1761096925.mov` | 23 MB | ~15s | Unknown | ✅ Clean! (UV fix applied) |

**Recommendation:** Recapture video #1 with the UV fix for consistent quality.

---

## 🎯 Training Data Requirements

### Minimum (For Initial Training):
- **Target:** 10 videos
- **Current:** 2 videos
- **Remaining:** 8 videos needed

### Recommended (For Good Quality):
- **Target:** 15 videos
- **Current:** 2 videos
- **Remaining:** 13 videos needed

### Optimal (For Production):
- **Target:** 20-30 videos
- **Current:** 2 videos
- **Remaining:** 18-28 videos

---

## 📋 Suggested Capture Plan

### Week 1 - Capture Sessions:

#### Session 1: Indoor Scenes (30 min)
- [ ] Living room - furniture at different depths
- [ ] Kitchen - counters and appliances
- [ ] Bedroom - bed, closet, varied lighting
- [ ] Hallway/stairs - depth variation
- [ ] Office/desk - computer, books, objects

**Goal:** 5 videos, ~10-20 seconds each

#### Session 2: Outdoor Scenes (30 min)
- [ ] Building exteriors - architecture
- [ ] Trees/nature - varied depth
- [ ] Street scene - cars, people
- [ ] Park/garden - natural objects
- [ ] Sky/horizon - distance variation

**Goal:** 5 videos, ~10-20 seconds each

#### Session 3: Object Focus (20 min)
- [ ] Close-up objects (phone, cup, book)
- [ ] Person at different distances
- [ ] Moving objects (walking, car)
- [ ] Textured surfaces (wall, fabric, wood)
- [ ] Mixed lighting (bright/shadow)

**Goal:** 5 videos, ~10-20 seconds each

---

## 🎬 Recording Tips

### For Best Training Data:

1. **Variety is Key:**
   - Different scenes (not just similar rooms)
   - Different depth ranges (near/far)
   - Different lighting conditions
   - Different textures and materials

2. **Movement:**
   - Slow panning works best
   - Or stationary camera
   - Avoid fast motion (harder to learn)

3. **Duration:**
   - 10-30 seconds per clip
   - Longer is fine, but diminishing returns
   - Focus on diversity over length

4. **Quality Checks:**
   - Extract depth frames to verify no green artifacts
   - Check that shapes are visible
   - Ensure good lighting (not too dark)

---

## 🔬 Quick Quality Check

After capturing each video:

```bash
# Extract 1 depth frame to verify
ffmpeg -i ~/Downloads/lumaflow_NEW.mov -map 0:1 -vframes 1 \
  ~/Downloads/check_depth.png -y

# Open it
open ~/Downloads/check_depth.png
```

**Look for:**
- ✅ Pure grayscale (no green/purple)
- ✅ Shapes clearly visible
- ✅ Depth gradient present
- ✅ Good contrast

**If all checks pass → Move to training folder!**

---

## 📁 File Management

### After Each Capture:
```bash
# Move to training folder
mv ~/Downloads/lumaflow_*.mov ~/lumaflow_training_data/

# Check total
ls -lh ~/lumaflow_training_data/*.mov
```

### Training Folder Location:
```
~/lumaflow_training_data/
├── lumaflow_1761095137.mov (19 MB) ⚠️ Old (has green artifacts)
├── lumaflow_1761096925.mov (23 MB) ✅ Clean!
└── ... (capture 8-13 more)
```

---

## 🚀 When Ready to Train

### Prerequisites:
- ✅ At least 10 videos captured
- ✅ All videos moved to `~/lumaflow_training_data/`
- ✅ Depth quality verified (no green artifacts)

### Next Steps:
1. **Test data loader:**
   ```bash
   cd /Users/yarontorbaty/Documents/Code/Aiv1-LumaFlow/generative_codec
   python data/iphone_loader.py
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start training:**
   ```bash
   python train.py --data_dir ~/lumaflow_training_data --epochs 50
   ```

---

## 💰 Cost Estimate

### Current Status:
- **Development:** $0 ✅
- **Capture time:** ~1 hour so far
- **Training:** Not started

### Projected:
- **Capture remaining videos:** ~1-2 hours
- **Training (AWS g4dn.xlarge):** ~$4-5 for 4-8 hours
- **Total estimated cost:** ~$5

---

## 📊 Progress Visualization

```
Training Data Collection Progress:
[██░░░░░░░░░░░░] 13% (2/15 videos)

Capture Sessions:
Indoor:   [ ] Not started
Outdoor:  [ ] Not started  
Objects:  [ ] Not started

Estimated completion: 1-2 hours of capture time remaining
```

---

## 🎯 Current Priorities

1. **📱 Capture 8 more videos (minimum)** to reach 10 total
2. **🎨 Focus on diversity** (indoor/outdoor/objects)
3. **✅ Verify depth quality** for each capture
4. **📊 Track progress** in this document

---

## ✅ Quality Milestones

- [x] iPhone app working
- [x] Depth capture functional
- [x] UV plane bug fixed
- [x] Clean grayscale depth verified
- [ ] 10 videos captured (minimum)
- [ ] 15 videos captured (recommended)
- [ ] Data loader tested
- [ ] Training started

---

**Next Action:** Capture 8-13 more videos across different scenes! 🎥

