# Quality Experiments Report

## Summary

Found **3 experiments** that met quality criteria (PSNR > 25 dB) and were under 10 Mbps:

| Experiment ID | Bitrate | PSNR | SSIM | Quality | Status |
|--------------|---------|------|------|---------|--------|
| `proc_exp_1760706922` | 7.05 Mbps | 29.86 dB | 0.8464 | ✅ Acceptable | ⚠️  No videos saved |
| `proc_exp_1760718728` | 2.70 Mbps | 25.48 dB | 0.7347 | ✅ Acceptable | ⚠️  No videos saved |
| `proc_exp_1760716815` | 2.54 Mbps | 25.31 dB | 0.7763 | ✅ Acceptable | ⚠️  No videos saved |

---

## ⚠️  Why No Reconstructed Videos or Decoder Code?

### Root Cause

The orchestrator only saves videos and decoder code when:
```python
target_achieved = bitrate_mbps < 1.0  # Very aggressive target!
```

**Current logic** in `procedural_experiment_runner.py:823`:
```python
if reconstructed_path and target_achieved:  # Only upload for successful experiments
    video_url = self._upload_reconstructed_video(reconstructed_path, experiment_id)

if decoder_code and target_achieved:
    decoder_s3_key = self._save_decoder_code(decoder_code, experiment_id)
```

### Impact

- **Target**: < 1.0 Mbps (extremely aggressive)
- **Best result**: 2.54 Mbps (2.5× the target)
- **Result**: No videos or decoders saved for ANY experiment

### Statistics

Out of **102 total experiments**:
- ✅ 3 have PSNR > 25 dB and bitrate < 10 Mbps (quality verified)
- ❌ 0 have bitrate < 1.0 Mbps (target achieved)
- ❌ 0 have saved reconstructed videos
- ❌ 0 have saved decoder code
- ❌ 0 have detailed LLM analysis (hypothesis, insights, generated code)

---

## Experiment Details

### 1. proc_exp_1760706922 (BEST QUALITY)

**Performance:**
- Bitrate: **7.05 Mbps**
- PSNR: **29.86 dB** (close to "good" quality threshold of 30 dB)
- SSIM: **0.8464** (high similarity)
- Quality: Acceptable
- File Size: 8.81 MB (10 seconds)
- Reduction vs HEVC: 29.5%

**What was done:**
- Compression experiment (minimal metadata)
- Quality verified: Yes ✅
- Video/decoder saved: No ❌ (bitrate > 1.0 Mbps)

**Technical Details:**
- Resolution: 1920x1080 (Full HD)
- Frame Rate: 30 FPS
- Total Frames: 300
- Duration: 10.0 seconds

---

### 2. proc_exp_1760718728

**Performance:**
- Bitrate: **2.70 Mbps**
- PSNR: **25.48 dB** (acceptable quality)
- SSIM: **0.7347** (moderate similarity)
- Quality: Acceptable
- File Size: 3.38 MB (10 seconds)
- Reduction vs HEVC: 73.0%

**What was done:**
- Compression experiment (minimal metadata)
- Quality verified: Yes ✅
- Video/decoder saved: No ❌ (bitrate > 1.0 Mbps)

**Technical Details:**
- Resolution: 1920x1080 (Full HD)
- Frame Rate: 30 FPS
- Total Frames: 300
- Duration: 10.0 seconds

---

### 3. proc_exp_1760716815 (BEST BITRATE)

**Performance:**
- Bitrate: **2.54 Mbps** (lowest bitrate!)
- PSNR: **25.31 dB** (acceptable quality)
- SSIM: **0.7763** (good similarity)
- Quality: Acceptable
- File Size: 3.17 MB (10 seconds)
- Reduction vs HEVC: 74.6%

**What was done:**
- Compression experiment (minimal metadata)
- Quality verified: Yes ✅
- Video/decoder saved: No ❌ (bitrate > 1.0 Mbps)

**Technical Details:**
- Resolution: 1920x1080 (Full HD)
- Frame Rate: 30 FPS
- Total Frames: 300
- Duration: 10.0 seconds

---

## Quality Metrics Explained

### PSNR (Peak Signal-to-Noise Ratio)

Measures reconstruction quality:
- **< 25 dB**: Poor quality (blocky/blurry)
- **25-30 dB**: Acceptable quality ← All 3 experiments
- **30-35 dB**: Good quality
- **> 35 dB**: Excellent quality (H.264/HEVC level)

**Our Results:**
- Best: 29.86 dB (very close to "good" threshold!)
- Worst: 25.31 dB (acceptable)

### SSIM (Structural Similarity Index)

Measures perceptual similarity (0-1 scale):
- **< 0.5**: Poor similarity
- **0.5-0.7**: Moderate similarity
- **0.7-0.9**: Good similarity ← All 3 experiments
- **> 0.9**: Excellent similarity

**Our Results:**
- Best: 0.8464 (good)
- Worst: 0.7347 (good)

---

## Why Minimal Metadata?

The experiment data lacks detailed information because:

1. **No detailed LLM analysis saved** (hypothesis, insights, root cause)
   - This data should exist but isn't being stored
   
2. **No generated code saved** (compress/decompress functions)
   - The code exists during execution but isn't persisted
   
3. **No approach description** (only "Compression experiment")
   - The orchestrator isn't saving the detailed approach

This suggests the **analysis phase** might not be running or saving data properly for experiments that don't achieve the < 1.0 Mbps target.

---

## Reconstructed Videos

### Where Are They?

The `adaptive_codec_agent.py` creates reconstructed videos during quality verification:

```python
# Line 358-363 in adaptive_codec_agent.py
output_reconstructed = f"/tmp/reconstructed_{timestamp}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_reconstructed, fourcc, video_fps, (resolution[0], resolution[1]))
for frame in reconstructed_frames:
    out.write(frame)
out.release()
```

**Problem**: These videos are created in `/tmp/` but:
1. Only uploaded to S3 if `target_achieved = True` (bitrate < 1.0 Mbps)
2. Deleted after execution phase
3. Never accessible to users

### Can We Recover Them?

❌ No - the reconstructed videos were temporary and deleted after quality measurement.

To get reconstructed videos, we need to either:
1. **Re-run** the experiments (requires the original LLM-generated code)
2. **Change the target** to < 10 Mbps so future experiments save videos
3. **Always save videos** regardless of target achievement

---

## Recommendations

### Option 1: Lower the Target (Recommended)

Change the target from 1.0 Mbps to 10.0 Mbps:

```python
# src/agents/procedural_experiment_runner.py:823
target_achieved = bitrate_mbps < 10.0  # More realistic target
```

**Impact**: Future experiments with bitrate < 10 Mbps will save videos and decoders.

### Option 2: Always Save Videos (Best for Research)

Remove the target requirement:

```python
# Always save videos for quality-verified experiments
if reconstructed_path and real_metrics.get('quality_verified'):
    video_url = self._upload_reconstructed_video(reconstructed_path, experiment_id)
```

**Impact**: All experiments with quality metrics get videos saved (better for analysis).

### Option 3: Save Detailed Analysis Always

Modify the orchestrator to always save LLM analysis (hypothesis, insights, code):

```python
# Save detailed analysis regardless of target
experiments_array = [{
    'experiment_type': 'real_procedural_generation',
    'status': 'completed',
    'approach': approach,
    'hypothesis': hypothesis,  # Add these
    'insights': insights,
    'root_cause': root_cause,
    'generated_code': generated_code,
    'real_metrics': real_metrics,
    # ...
}]
```

**Impact**: Better understanding of what each experiment actually tried.

---

## Next Steps

To get reconstructed videos and detailed analysis:

### 1. Update the Target Threshold

```bash
# Edit src/agents/procedural_experiment_runner.py
# Line 823: Change target_achieved = bitrate_mbps < 1.0
#        To: target_achieved = bitrate_mbps < 10.0
```

### 2. Re-run Experiments

Let the orchestrator generate new experiments. With the new threshold, experiments achieving < 10 Mbps will have:
- ✅ Reconstructed videos (S3 + CloudFront URLs)
- ✅ Decoder code (downloadable)
- ✅ Detailed LLM analysis
- ✅ Full approach description

### 3. Monitor Dashboard

Check for:
- Video URLs appearing in experiment data
- Decoder links becoming available
- Better metadata in blog posts

---

## Current System Status

### What's Working ✅

1. **Quality verification**: PSNR and SSIM are being calculated correctly
2. **Video generation**: Procedural video generation is working (1920x1080, 30fps)
3. **Compression**: Experiments are compressing videos
4. **Decompression**: Frames are being reconstructed for quality measurement
5. **Metrics collection**: Bitrate, file size, duration all accurate

### What's Not Working ❌

1. **Video persistence**: Reconstructed videos not saved (too aggressive target)
2. **Decoder persistence**: Decoder code not saved (same reason)
3. **Metadata completeness**: Missing hypothesis, insights, detailed approach
4. **Analysis phase**: Not saving full LLM analysis to database

### Quick Fix

The fastest way to start getting usable results:

```bash
cd /Users/yarontorbaty/Documents/Code/AiV1

# Update target threshold
sed -i '' 's/bitrate_mbps < 1.0/bitrate_mbps < 10.0/' src/agents/procedural_experiment_runner.py

# Commit changes
git add src/agents/procedural_experiment_runner.py
git commit -m "Lower video save threshold to 10 Mbps for better artifact collection"
git push origin main
```

Then restart the orchestrator and wait for new experiments that meet the < 10 Mbps criteria.

---

## Summary Table

| Metric | Experiment 1 | Experiment 2 | Experiment 3 | Target | Status |
|--------|--------------|--------------|--------------|--------|--------|
| **Bitrate** | 7.05 Mbps | 2.70 Mbps | **2.54 Mbps** | < 1.0 Mbps | ❌ None meet target |
| **PSNR** | **29.86 dB** | 25.48 dB | 25.31 dB | > 25 dB | ✅ All pass |
| **SSIM** | **0.8464** | 0.7347 | 0.7763 | > 0.7 | ✅ All pass |
| **Quality** | Acceptable | Acceptable | Acceptable | Good | ⚠️  Close |
| **Video Saved** | ❌ | ❌ | ❌ | N/A | ❌ Target too low |
| **Decoder Saved** | ❌ | ❌ | ❌ | N/A | ❌ Target too low |

---

## Conclusion

You have **3 successful experiments** with:
- ✅ Quality verification (PSNR 25-30 dB, SSIM 0.73-0.85)
- ✅ Bitrate < 10 Mbps (2.54-7.05 Mbps)
- ✅ Full HD resolution (1920x1080)
- ❌ But no reconstructed videos or decoder code saved

**Reason**: The < 1.0 Mbps target is too aggressive. Best experiment achieved 2.54 Mbps (2.5× the target).

**Solution**: Update the target to < 10.0 Mbps so future experiments save artifacts for analysis.

