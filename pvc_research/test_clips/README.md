# PVC Test Clips

Drop your 3 anime clips here!

## 📋 Recommended Format

- **Resolution:** 720p or 1080p
- **Duration:** 5-15 seconds
- **FPS:** 24-30
- **Format:** MP4, MKV, or MOV

## 🎬 Good Test Content

### **Clip 1: Simple Animation**
- Few characters
- Clean backgrounds
- Solid colors
- Minimal motion
- **Expected:** 95%+ compression

### **Clip 2: Typical Anime Scene**
- Multiple characters
- Action sequences
- Some effects
- Moderate complexity
- **Expected:** 90%+ compression

### **Clip 3: Complex Scene**
- Detailed backgrounds
- Fast motion
- Visual effects
- High complexity
- **Expected:** 85-90% compression

## 📊 After Dropping Files

Run the batch test script:

```bash
cd /Users/yarontorbaty/Documents/Code/AiV1
python pvc_research/experiments/batch_test.py
```

This will:
1. Encode all 3 clips with PVC
2. Generate AV1 baselines for comparison
3. Decode and evaluate quality
4. Create a comparison report

## 📁 Output Structure

After testing:
```
test_clips/
├── anime_clip_1.mp4
├── anime_clip_2.mp4
└── anime_clip_3.mp4

experiments/
├── results/
│   ├── clip_1_scene.json
│   ├── clip_1_reconstructed.mp4
│   ├── clip_2_scene.json
│   ├── clip_2_reconstructed.mp4
│   ├── clip_3_scene.json
│   └── clip_3_reconstructed.mp4
├── baselines/
│   ├── clip_1_av1.mp4
│   ├── clip_2_av1.mp4
│   └── clip_3_av1.mp4
└── report.json (comparison results)
```

## 🎯 What We're Testing

- **Compression ratio** vs AV1
- **Quality metrics** (PSNR, SSIM)
- **Encoding speed**
- **Content type suitability**

---

**Drop your clips here and run the batch test!**

