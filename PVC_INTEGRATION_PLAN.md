# PVC + Neural Codec Integration Summary

## 🎯 System Overview

The AiV1 framework now supports **two parallel research tracks**:

### **1. Neural Codec (Existing)**
- **Location:** `v3/worker`, `v3/orchestrator`
- **Approach:** LLM-generated neural compression code
- **Target:** General-purpose video (HD frames)
- **Baseline:** HEVC 10Mbps
- **Metrics:** PSNR, SSIM vs SOURCE
- **Table:** `ai-codec-v3-experiments`
- **Status:** ✅ **Running** (5 experiments in progress as of Oct 19, 2025)

### **2. Procedural Codec (New)**
- **Location:** `pvc_research/`
- **Approach:** Demoscene-inspired procedural generation
- **Target:** Animation, anime, stylized content
- **Baseline:** AV1 5Mbps
- **Metrics:** PSNR, SSIM, VMAF vs original
- **Table:** `ai-codec-pvc-experiments` (to be created)
- **Status:** 🚧 **Ready for Deployment**

---

## 📊 Dashboard Integration

### **Option 1: Unified Dashboard (Recommended)**

Add a top-level tab switcher to the existing dashboard:

```
┌─────────────────────────────────────┐
│ AiV1 Video Codec Research v3.0      │
├─────────────────────────────────────┤
│ [Neural Codec] [Procedural Codec]   │  ← New tab switcher
├─────────────────────────────────────┤
│                                     │
│ (Existing neural codec experiments) │
│                                     │
└─────────────────────────────────────┘
```

**Benefits:**
- Single dashboard URL
- Easy comparison between approaches
- Unified user experience

**Implementation:**
- Add `PVC_DYNAMODB_TABLE` environment variable
- Modify `lambda_handler` to accept `?codec=neural` or `?codec=pvc`
- Add codec selector to header
- Fetch from appropriate table

### **Option 2: Separate Dashboard**

Create a dedicated PVC dashboard Lambda:

```
Neural Codec: aiv1codec.com/
PVC Dashboard: aiv1codec.com/pvc
```

**Benefits:**
- Complete separation of concerns
- Independent styling/features
- Easier to maintain separately

**Implementation:**
- Create `pvc_dashboard.py` Lambda
- Add CloudFront behavior for `/pvc/*`
- Duplicate dashboard structure

---

## 🏗️ AWS Infrastructure

### **What's Needed:**

1. **DynamoDB Table** (5 minutes)
   ```bash
   aws cloudformation create-stack \
     --stack-name pvc-database \
     --template-body file://pvc_research/infrastructure/pvc_database.yaml \
     --region us-east-1
   ```

2. **EC2 Worker Instance** (10 minutes)
   - Instance type: `t3.large` (animation processing doesn't need GPU)
   - AMI: Amazon Linux 2
   - Python 3.9+, OpenCV, FFmpeg
   - Install PVC requirements
   - Start HTTP worker

3. **Dashboard Integration** (15 minutes)
   - Option 1: Update existing Lambda
   - Option 2: Create new Lambda

### **Total Deployment Time: ~30 minutes**

---

## 📁 Data Schema

### **PVC Experiment Record:**

```json
{
  "experiment_id": "pvc_exp_1729384752",
  "timestamp": 1729384752,
  "status": "success",
  "codec_type": "procedural",
  "metadata": {
    "input_video": "anime_sample.mp4",
    "duration_sec": 10.0,
    "frame_count": 300,
    "resolution": [1280, 720]
  },
  "encoding_result": {
    "scene_size_bytes": 65536,
    "contour_count": 45,
    "object_count": 12,
    "texture_types": ["perlin", "worley", "solid"],
    "encoding_time_sec": 45.2
  },
  "quality_metrics": {
    "psnr_db": 32.4,
    "ssim": 0.89,
    "vmaf": 85.3
  },
  "compression_metrics": {
    "pvc_bitrate_mbps": 0.52,
    "baseline_bitrate_mbps": 5.0,
    "reduction_percent": 89.6,
    "compression_ratio": 9.6
  },
  "artifacts": {
    "scene_json_s3": "s3://ai-codec-v3-artifacts/pvc/exp_1729384752/scene.json",
    "reconstructed_video_s3": "s3://ai-codec-v3-artifacts/pvc/exp_1729384752/reconstructed.mp4",
    "original_video_s3": "s3://ai-codec-v3-artifacts/pvc/test_clips/anime_sample.mp4"
  }
}
```

---

## 🔄 Workflow Comparison

### **Neural Codec Workflow:**
```
Orchestrator → LLM generates code → Worker executes
                                      ↓
                                   Encode/Decode
                                      ↓
                                   Calculate metrics
                                      ↓
                                   Store in DynamoDB
```

### **PVC Workflow:**
```
Worker receives video → Extract contours
                         ↓
                      Track motion
                         ↓
                      Assign textures
                         ↓
                      Generate ISP (JSON)
                         ↓
                      Render reconstruction
                         ↓
                      Calculate metrics
                         ↓
                      Store in DynamoDB
```

**Key Difference:** No LLM needed for PVC! The algorithms are deterministic.

---

## 🎯 Next Steps (Recommended Order)

### **Phase 1: Local Testing** (Current)
✅ PVC code complete  
⬜ Test encoder/decoder locally  
⬜ Generate sample anime clip  
⬜ Verify metrics are reasonable  

### **Phase 2: AWS Deployment** (Next)
⬜ Create PVC DynamoDB table  
⬜ Launch EC2 worker instance  
⬜ Deploy PVC code to worker  
⬜ Run first experiment  

### **Phase 3: Dashboard Integration** (After)
⬜ Update dashboard Lambda (Option 1)  
⬜ Add codec selector  
⬜ Test both experiment types display  

### **Phase 4: Automation** (Final)
⬜ Automated test suite  
⬜ Batch processing  
⬜ Comparison reports  

---

## 💡 Recommended Approach for Now

Given that:
1. Neural codec is actively running experiments
2. PVC is a completely different approach
3. We want to avoid breaking the current system

**I recommend:**

1. **Deploy PVC infrastructure separately** (DynamoDB + EC2)
2. **Run initial PVC experiments manually** to validate the approach
3. **Create a simple comparison page** showing both results side-by-side
4. **Once validated, integrate into main dashboard** with a tab switcher

This incremental approach minimizes risk while allowing parallel development.

---

## 📊 Expected Results

### **Neural Codec:**
- **Target:** Match HEVC quality at lower bitrate
- **Current best:** TBD (experiments running)
- **Approach:** Learned compression via LLM

### **PVC:**
- **Target:** 90% bitrate reduction vs AV1 on animation
- **Expected:** 0.5 Mbps vs 5 Mbps baseline
- **Approach:** Geometric + procedural reconstruction

### **Together:**
- Neural codec for general video
- PVC for animation/stylized content
- Best of both worlds! 🎉

---

**Created:** October 19, 2025  
**Status:** Integration Plan Ready  
**Next Action:** Deploy PVC infrastructure or test locally first?

