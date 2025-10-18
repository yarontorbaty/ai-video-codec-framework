# 🔄 Experiments Restarted with Real Source Files

**Date:** October 18, 2025 - 9:30 AM EST  
**Status:** ✅ RUNNING (10 iterations)

---

## 🗑️ What Was Purged

### 1. DynamoDB Table
- **Table:** `ai-codec-v3-experiments`
- **Deleted:** 10 experiments (all previous results)
- **Current:** 0 experiments
- **Keys:** `experiment_id` (HASH) + `timestamp` (RANGE)

### 2. S3 Artifacts
- **Bucket:** `ai-codec-v3-artifacts-580473065386`
- **Deleted:**
  - `videos/` directory (7 files removed)
  - `decoders/` directory (all files removed)
- **Kept:**
  - `reference/source.mp4` (710MB) ✅
  - `reference/hevc_baseline.mp4` (12MB) ✅

### 3. Orchestrator
- **Stopped:** Previous running instance
- **Deployed:** Fresh code
- **Configured:** 10 iteration limit
- **Started:** New experiment run

---

## 🚀 New Experiment Configuration

### Orchestrator Settings
- **Max Iterations:** 10 (configured in `config.py`)
- **Worker URL:** http://10.0.2.10:8080
- **DynamoDB Table:** ai-codec-v3-experiments
- **LLM:** Claude via Anthropic API
- **Status:** ✅ Running (started iteration 1)

### Worker Configuration
- **Source Video:** Downloads from S3
  - **Key:** `reference/source.mp4`
  - **File:** SOURCE_HD_RAW.mp4 (710MB)
  - **Resolution:** 1920x1080 HD
- **Fallback:** Generates 640x480 test video if S3 fails
- **Status:** ✅ Ready

### Baseline Files
- **Source:** `test_data/SOURCE_HD_RAW.mp4` (710MB) → S3
- **HEVC:** `test_data/HEVC_HD_10Mbps.mp4` (12MB) → S3
- **Purpose:** Consistent baseline for all experiments

---

## 📊 Experiment Flow

```
1. Orchestrator (Iteration 1-10)
   ↓
2. LLM generates compression code
   ↓
3. Code sent to Worker
   ↓
4. Worker downloads 710MB source from S3
   ↓
5. Worker executes encoding/decoding
   ↓
6. Metrics calculated (PSNR/SSIM)
   ↓
7. Results saved to DynamoDB
   ↓
8. Artifacts uploaded to S3:
   - Reconstructed video
   - Decoder code
   ↓
9. Dashboard displays results
   ↓
10. Repeat for next iteration
```

---

## 🎯 Expected Results

### After 10 Iterations

**DynamoDB:**
- 10 experiment records
- Each with:
  - experiment_id
  - iteration (1-10)
  - status (success/failed)
  - metrics (PSNR, SSIM, bitrate, compression)
  - artifacts (video_url, decoder_s3_key)
  - llm_reasoning

**S3:**
- `videos/` directory with up to 10 reconstructed videos
- `decoders/` directory with up to 10 decoder Python files

**Dashboard:**
- Up to 10 experiments displayed
- Separated into Successful/Failed tabs
- Quality badges and tier achievements
- Links to videos and decoder code

---

## 🔍 Monitoring

### Check Progress
```bash
# Count experiments
aws dynamodb scan --table-name ai-codec-v3-experiments --select COUNT --query 'Count' --output text

# View orchestrator logs
aws ssm send-command --region us-east-1 \
  --instance-ids i-00d8ebe7d25026fdd \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["tail -50 /home/ec2-user/orchestrator/orchestrator.log"]' \
  --output text --query 'Command.CommandId'

# Then get output:
aws ssm get-command-invocation --region us-east-1 \
  --command-id <COMMAND_ID> \
  --instance-id i-00d8ebe7d25026fdd \
  --query 'StandardOutputContent' --output text
```

### Watch Dashboard
- **CloudFront:** https://d3sbni9ahh3hq.cloudfront.net
- **Domain (after DNS):** https://aiv1codec.com
- **Refresh:** Every 60 seconds (cache TTL)

---

## 📈 Improvements from Previous Run

### Source Material
| Before | After |
|--------|-------|
| Generated 2-sec test (640x480) | Real 710MB HD video (1920x1080) |
| Synthetic patterns | Actual video content |
| Not representative | Production-quality footage |

### Consistency
| Before | After |
|--------|-------|
| Different video each run | Same source every time |
| Hard to compare | Reproducible results |
| Random patterns | Consistent baseline |

### Quality Metrics
| Before | After |
|--------|-------|
| Metrics on synthetic video | Metrics on real HD footage |
| Not representative | Realistic quality assessment |
| Lower resolution | Full HD resolution |

---

## ⏱️ Timeline

### Startup
- **9:24 AM** - DynamoDB purged (10 experiments deleted)
- **9:24 AM** - S3 artifacts cleared
- **9:24 AM** - Orchestrator code deployed
- **9:25 AM** - Orchestrator started
- **9:25 AM** - Iteration 1 began

### Expected Completion
- **Each iteration:** ~5-10 minutes
- **Total time:** ~50-100 minutes
- **Completion:** ~10:30-11:00 AM EST

---

## 🎨 Dashboard Features

All features from v3.0 dashboard:
- ✅ Dark theme
- ✅ Font Awesome icons
- ✅ Tabbed interface (Successful/Failed)
- ✅ Quality badges
- ✅ Achievement tiers (Gold/Silver/Bronze)
- ✅ Full blog posts
- ✅ Download buttons (video/decoder)
- ✅ Pagination (10 per page)
- ✅ LLM project summary
- ✅ Sidebar navigation
- ✅ Reference video links (source & HEVC)

---

## 🔧 Technical Details

### Orchestrator
- **Instance:** i-00d8ebe7d25026fdd
- **Location:** /home/ec2-user/orchestrator
- **Files:**
  - main.py
  - config.py
  - llm_client_simple.py
  - experiment_manager.py
- **Log:** orchestrator.log

### Worker
- **Instance:** i-01113a08e8005b235
- **Location:** /home/ec2-user/worker
- **Port:** 8080
- **Files:**
  - main.py
  - experiment_runner.py
  - metrics_calculator.py
  - s3_uploader.py

### Infrastructure
- **Region:** us-east-1
- **DynamoDB:** ai-codec-v3-experiments
- **S3:** ai-codec-v3-artifacts-580473065386
- **Lambda:** ai-codec-v3-dashboard
- **CloudFront:** E3PUY7OMWPWSUN
- **Domain:** aiv1codec.com

---

## 🎯 Success Criteria

For this run to be considered successful:

1. ✅ **All 10 iterations complete** without crashes
2. ✅ **Source video downloads** from S3 for each experiment
3. ✅ **LLM generates valid code** for compression/decompression
4. ✅ **Metrics are calculated** (PSNR, SSIM, bitrate, compression)
5. ✅ **Results saved to DynamoDB** with all fields
6. ✅ **Artifacts uploaded to S3** (videos and decoders)
7. ✅ **Dashboard displays** all experiments with quality badges
8. ✅ **Downloads work** (video and decoder files)

---

## 📝 Verification Steps

After completion:

```bash
# 1. Check experiment count (should be 10)
aws dynamodb scan --table-name ai-codec-v3-experiments --select COUNT

# 2. Verify S3 videos
aws s3 ls s3://ai-codec-v3-artifacts-580473065386/videos/ --recursive

# 3. Verify S3 decoders
aws s3 ls s3://ai-codec-v3-artifacts-580473065386/decoders/ --recursive

# 4. Check dashboard
curl -s https://d3sbni9ahh3hq.cloudfront.net/ | grep "AiV1"

# 5. Verify source video usage
aws ssm send-command --region us-east-1 \
  --instance-ids i-01113a08e8005b235 \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["grep \"Source video downloaded\" /home/ec2-user/worker/*.log | tail -10"]'
```

---

## 🚨 Troubleshooting

### If orchestrator stops
```bash
# Restart
aws ssm send-command --region us-east-1 \
  --instance-ids i-00d8ebe7d25026fdd \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["cd /home/ec2-user/orchestrator","nohup python3 main.py > orchestrator.log 2>&1 &"]'
```

### If worker stops
```bash
# Restart
aws ssm send-command --region us-east-1 \
  --instance-ids i-01113a08e8005b235 \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["cd /home/ec2-user/worker","nohup python3 main.py > worker.log 2>&1 &"]'
```

### If S3 download fails
- Worker will automatically fall back to generated test video
- Check IAM role permissions on worker instance
- Verify S3 bucket and key exist

---

## 📊 Comparison: Old vs New

### Old Experiments (Purged)
- **Count:** 10
- **Source:** Generated synthetic video
- **Resolution:** 640x480
- **Duration:** 2 seconds
- **Quality:** Not representative
- **Results:** Varied, hard to compare

### New Experiments (Running)
- **Count:** 10 (in progress)
- **Source:** Real HD video from S3
- **Resolution:** 1920x1080
- **Duration:** Full video length
- **Quality:** Production-grade
- **Results:** Consistent, comparable

---

## 🎊 Summary

✅ **Purged:** All old experiments and artifacts  
✅ **Deployed:** Orchestrator with 10 iteration limit  
✅ **Configured:** Worker to use 710MB source video  
✅ **Started:** New experiment run  
✅ **Monitoring:** Dashboard and logs available  
✅ **Expected:** 10 high-quality experiments in ~1 hour  

**Status:** RUNNING  
**Progress:** Iteration 1/10  
**Dashboard:** https://d3sbni9ahh3hq.cloudfront.net  
**Domain:** https://aiv1codec.com (pending DNS)

---

*Started: October 18, 2025 at 9:25 AM EST*  
*Expected Completion: ~10:30-11:00 AM EST*

