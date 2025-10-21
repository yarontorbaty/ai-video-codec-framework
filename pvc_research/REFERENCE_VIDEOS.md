# PVC Reference Videos - Anime Test Clips

**Created:** October 19-20, 2025  
**S3 Location:** `s3://ai-codec-v3-artifacts-580473065386/pvc/reference/`

---

## 📊 **Clip Details**

### **Clip 1: source_anime_01.mp4**
- **Duration:** 5.5 seconds
- **Source Size:** 6.1 MB (9.26 Mbps)
- **AV1 Size:** 4.1 MB (6.27 Mbps)
- **AV1 Reduction:** 40% vs source
- **S3 Paths:**
  - Source: `s3://ai-codec-v3-artifacts-580473065386/pvc/reference/source_anime_01.mp4`
  - AV1: `s3://ai-codec-v3-artifacts-580473065386/pvc/reference/av1_anime_01.mp4`

### **Clip 2: source_anime_02.mp4**
- **Duration:** 5.8 seconds
- **Source Size:** 8.6 MB (12.38 Mbps)
- **AV1 Size:** 4.5 MB (6.52 Mbps)
- **AV1 Reduction:** 50% vs source
- **S3 Paths:**
  - Source: `s3://ai-codec-v3-artifacts-580473065386/pvc/reference/source_anime_02.mp4`
  - AV1: `s3://ai-codec-v3-artifacts-580473065386/pvc/reference/av1_anime_02.mp4`

### **Clip 3: source_anime_03.mp4**
- **Duration:** 10.3 seconds
- **Source Size:** 18 MB (14.88 Mbps)
- **AV1 Size:** 5.4 MB (4.37 Mbps)
- **AV1 Reduction:** 80% vs source
- **S3 Paths:**
  - Source: `s3://ai-codec-v3-artifacts-580473065386/pvc/reference/source_anime_03.mp4`
  - AV1: `s3://ai-codec-v3-artifacts-580473065386/pvc/reference/av1_anime_03.mp4`

---

## 🎯 **PVC Target**

PVC needs to achieve **90% compression vs AV1**, which means:

| Clip | AV1 Size | PVC Target (90% reduction) | Max Size |
|------|----------|---------------------------|----------|
| 1 | 4.1 MB | 90% smaller | **410 KB** |
| 2 | 4.5 MB | 90% smaller | **450 KB** |
| 3 | 5.4 MB | 90% smaller | **540 KB** |

---

## 📝 **Testing Plan**

### **Phase 1: Local Testing**
```bash
cd /Users/yarontorbaty/Documents/Code/AiV1
python pvc_research/experiments/batch_test.py
```

### **Phase 2: AWS Worker**
1. Deploy PVC worker (EC2 t3.large)
2. Cache these 6 files on worker at `/home/ec2-user/pvc/cache/`
3. Run experiments using cached files
4. Store results in `ai-codec-pvc-experiments` DynamoDB table

---

## 🔄 **Worker Cache Structure**

```
/home/ec2-user/pvc/cache/
├── source_anime_01.mp4
├── source_anime_02.mp4
├── source_anime_03.mp4
├── av1_anime_01.mp4
├── av1_anime_02.mp4
└── av1_anime_03.mp4
```

Worker will:
1. Download once from S3 on first use
2. Check S3 ETag on subsequent runs
3. Re-download only if changed
4. Use local cache for all experiments

---

## 📊 **Expected PVC Results**

Based on content analysis:

### **Clip 1 (Simple):**
- **Content:** Clean animation, solid colors, simple motion
- **Expected:** 95%+ compression vs AV1
- **Target Size:** <200 KB
- **Quality:** PSNR >30dB

### **Clip 2 (Medium):**
- **Content:** Typical anime, moderate detail
- **Expected:** 90-93% compression vs AV1
- **Target Size:** <400 KB
- **Quality:** PSNR >28dB

### **Clip 3 (Complex):**
- **Content:** More detail, longer duration
- **Expected:** 87-90% compression vs AV1
- **Target Size:** <500 KB
- **Quality:** PSNR >26dB

---

## 🎬 **Next Steps**

1. ✅ **Source clips provided** (3 anime clips)
2. ✅ **AV1 baselines created** (5 Mbps encoding)
3. ✅ **Uploaded to S3** (all 6 files)
4. ⬜ **Cache on PVC worker** (when deployed)
5. ⬜ **Run local tests** (verify PVC pipeline)
6. ⬜ **Deploy to AWS** (if local tests pass)
7. ⬜ **Integrate with dashboard** (show PVC experiments)

---

**All reference files ready for PVC testing!** 🚀

