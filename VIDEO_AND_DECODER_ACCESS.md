# ✅ Video Upload & Decoder Code Access - IMPLEMENTED

## Summary

Successfully implemented automatic video uploading and decoder code storage for **successful experiments** (< 1 Mbps target achieved).

---

## What Was Implemented

### 1. Video Upload to S3 📹

**Location:** `src/agents/procedural_experiment_runner.py`

**Function:** `_upload_reconstructed_video()`
```python
def _upload_reconstructed_video(self, video_path: str, experiment_id: str) -> Optional[str]:
    """Upload reconstructed video to S3 and return presigned URL (valid 7 days)"""
    s3 = boto3.client('s3')
    key = f"reconstructed/{experiment_id}.mp4"
    s3.upload_file(video_path, VIDEO_BUCKET, key)
    url = s3.generate_presigned_url('get_object', ..., ExpiresIn=604800)
    return url
```

**Features:**
- ✅ Uploads reconstructed video to `s3://ai-video-codec-videos-580473065386/reconstructed/`
- ✅ Generates presigned URL (valid for 7 days)
- ✅ Logs file size and upload status
- ✅ Only uploads for **successful experiments** (target_achieved = True)

### 2. Decoder Code Storage 💾

**Location:** `src/agents/procedural_experiment_runner.py`

**Function:** `_save_decoder_code()`
```python
def _save_decoder_code(self, decoder_code: str, experiment_id: str) -> Optional[str]:
    """Save decoder code to S3 for easy retrieval"""
    s3 = boto3.client('s3')
    key = f"decoders/{experiment_id}_decoder.py"
    
    code_with_header = f'''#!/usr/bin/env python3
"""
Decoder for experiment: {experiment_id}
Generated: {datetime.utcnow().isoformat()}Z
"""

{decoder_code}
'''
    
    s3.put_object(Bucket=VIDEO_BUCKET, Key=key, Body=code_with_header.encode('utf-8'))
    return key
```

**Features:**
- ✅ Saves decoder Python code to `s3://ai-video-codec-videos-580473065386/decoders/`
- ✅ Adds helpful header with experiment ID and timestamp
- ✅ Returns S3 key for retrieval
- ✅ Only saves for **successful experiments**

### 3. DynamoDB Storage 📊

**Location:** `src/agents/procedural_experiment_runner.py` - `_phase_analysis()`

**Stored Fields:**
```json
{
  "experiments": [
    {
      "experiment_type": "real_procedural_generation",
      "status": "completed",
      "real_metrics": {...},
      "comparison": {...},
      "video_url": "https://s3.amazonaws.com/...?X-Amz-Signature=...",
      "decoder_s3_key": "decoders/proc_exp_1760710000_decoder.py"
    }
  ]
}
```

**Logic:**
- Only uploads video/decoder if `target_achieved == True` (bitrate < 1.0 Mbps)
- Stores presigned video URL (expires in 7 days)
- Stores decoder S3 key (permanent, can regenerate URL)

### 4. Dashboard Display 🎨

**Location:** `lambda/index_ssr.py` - Blog page

**Video Section:**
```html
<div class="blog-section" style="background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%); border-left: 4px solid #667eea;">
  <h3><i class="fas fa-video"></i> Reconstructed Video</h3>
  <p>View the decompressed video output from this experiment:</p>
  <a href="{video_url}" target="_blank" style="...">
    <i class="fas fa-play-circle"></i> View Reconstructed Video
  </a>
  <p style="font-size: 0.85em; color: #666;">
    <i class="fas fa-info-circle"></i> Presigned URL expires in 7 days. 
    Video shows the decoded output after compression.
  </p>
</div>
```

**Decoder Section:**
```html
<div class="blog-section" style="background: #f0f9ff; border-left: 4px solid #0ea5e9;">
  <h3><i class="fas fa-code"></i> Decoder Code</h3>
  <p>Download the Python decoder used to reconstruct frames from compressed data:</p>
  <a href="https://ai-video-codec-videos-580473065386.s3.amazonaws.com/{decoder_s3_key}" target="_blank" style="...">
    <i class="fas fa-download"></i> Download Decoder (.py)
  </a>
  <p style="font-size: 0.85em; color: #666;">
    <i class="fas fa-code-branch"></i> Use this decoder to reconstruct video frames 
    from the compressed data format.
  </p>
</div>
```

**Features:**
- ✅ Only shows sections if video_url / decoder_s3_key exist
- ✅ Beautiful gradient backgrounds with icons
- ✅ Clear call-to-action buttons
- ✅ Informative descriptions

---

## How It Works

### Experiment Flow

```
┌──────────────────────────────────┐
│  1. DESIGN PHASE                 │
│     LLM generates compression    │
│     algorithm                    │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  2. VALIDATION PHASE             │
│     Validate code syntax         │
│     Check security               │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  3. DEPLOYMENT PHASE             │
│     Save code to disk            │
│     Prepare for execution        │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  4. EXECUTION PHASE              │
│     Run compression on frames    │
│     Save compressed data         │
│     Save reconstructed video     │ ← NEW!
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  5. VERIFICATION PHASE           │
│     Calculate PSNR/SSIM          │
│     Verify quality metrics       │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  6. ANALYSIS PHASE               │
│     Calculate bitrate reduction  │
│     Check if target achieved     │
│                                  │
│  IF target_achieved (< 1 Mbps):  │
│    ✅ Upload video to S3         │ ← NEW!
│    ✅ Save decoder code to S3    │ ← NEW!
│    ✅ Store URLs in DynamoDB     │ ← NEW!
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  7. DASHBOARD DISPLAY            │
│     Show "View Video" button     │ ← NEW!
│     Show "Download Decoder"      │ ← NEW!
└──────────────────────────────────┘
```

---

## Example: Successful Experiment

### Experiment Details
- **ID:** `proc_exp_1760710000`
- **Bitrate:** 0.73 Mbps (< 1.0 Mbps target ✅)
- **PSNR:** 19.84 dB (poor quality)
- **SSIM:** 0.68 (acceptable)
- **Status:** Completed

### What Gets Stored

**S3 Objects:**
1. **Video:** `s3://ai-video-codec-videos-580473065386/reconstructed/proc_exp_1760710000.mp4`
   - Size: ~1-10 MB (depending on compression)
   - Format: MP4 (H.264)
   - Presigned URL valid for 7 days

2. **Decoder:** `s3://ai-video-codec-videos-580473065386/decoders/proc_exp_1760710000_decoder.py`
   - Contains: `decompress_video_frame()` function
   - Includes: Header with experiment ID and timestamp
   - Format: Python source code

**DynamoDB:**
```json
{
  "experiment_id": "proc_exp_1760710000",
  "timestamp": 1760710000,
  "status": "completed",
  "experiments": "[{
    \"experiment_type\": \"real_procedural_generation\",
    \"status\": \"completed\",
    \"real_metrics\": {
      \"bitrate_mbps\": 0.73,
      \"psnr_db\": 19.84,
      \"ssim\": 0.68,
      \"quality\": \"poor\"
    },
    \"comparison\": {
      \"hevc_baseline_mbps\": 10.0,
      \"reduction_percent\": 92.7,
      \"target_achieved\": true
    },
    \"video_url\": \"https://ai-video-codec-videos-580473065386.s3.amazonaws.com/reconstructed/proc_exp_1760710000.mp4?...\",
    \"decoder_s3_key\": \"decoders/proc_exp_1760710000_decoder.py\"
  }]"
}
```

### Dashboard Display

User sees on `/blog.html`:

```
┌─────────────────────────────────────────────────┐
│ Iteration 42: Parameter Storage Optimization   │
│ ✅ Completed | 2025-10-17                       │
├─────────────────────────────────────────────────┤
│                                                 │
│ 📊 Results                                      │
│ ╔════════╦═══════════╦════════╗                 │
│ ║ 0.73   ║ -92.7%    ║ 19.8dB ║                 │
│ ║ Mbps   ║ vs HEVC   ║ PSNR   ║                 │
│ ╚════════╩═══════════╩════════╝                 │
│                                                 │
│ 🎬 Reconstructed Video                          │
│ View the decompressed video output:             │
│ ┌───────────────────────────────┐               │
│ │ ▶ View Reconstructed Video    │               │
│ └───────────────────────────────┘               │
│ ⓘ Presigned URL expires in 7 days              │
│                                                 │
│ 💻 Decoder Code                                 │
│ Download the Python decoder:                    │
│ ┌───────────────────────────────┐               │
│ │ ⬇ Download Decoder (.py)      │               │
│ └───────────────────────────────┘               │
│ 🔧 Use this to reconstruct frames               │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## Storage Costs

### Per Experiment

**Video File:**
- Typical size: 1-10 MB (compressed output)
- S3 Standard storage: ~$0.023/GB/month
- Cost per video: **~$0.0002/month**

**Decoder File:**
- Typical size: 5-50 KB (Python code)
- Cost per decoder: **~$0.000001/month** (negligible)

**Presigned URL:**
- No storage cost (generated on-demand)
- Expires after 7 days (security)

### Annual Estimates

**Scenario 1: 100 successful experiments/year**
- Videos: 100 × 5 MB = 500 MB
- Decoders: 100 × 20 KB = 2 MB
- Total storage: ~502 MB
- Annual cost: 502 MB × $0.023/GB × 12 months = **$0.14/year**

**Scenario 2: 1000 successful experiments/year**
- Videos: 1000 × 5 MB = 5 GB
- Decoders: 1000 × 20 KB = 20 MB
- Total storage: ~5.02 GB
- Annual cost: 5.02 GB × $0.023/GB × 12 months = **$1.39/year**

**Verdict:** Extremely affordable! ✅

---

## Security & Access Control

### Presigned URLs
- ✅ Time-limited (7 days expiry)
- ✅ No public bucket access required
- ✅ URLs are cryptographically signed
- ✅ Revocable by deleting S3 object

### Decoder Code
- ✅ Direct S3 URLs (public read via bucket policy)
- ✅ No sensitive data in decoder code
- ✅ Python source code (inspectable)

### S3 Bucket Policy
Ensure bucket allows public read for decoders:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::ai-video-codec-videos-580473065386/decoders/*"
    }
  ]
}
```

---

## Usage Examples

### Download and Use Decoder

```bash
# Download decoder
curl -O "https://ai-video-codec-videos-580473065386.s3.amazonaws.com/decoders/proc_exp_1760710000_decoder.py"

# Inspect code
cat proc_exp_1760710000_decoder.py

# Use in Python
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from proc_exp_1760710000_decoder import decompress_video_frame

# Load compressed data
with open('compressed_data.bin', 'rb') as f:
    compressed = f.read()

# Decompress frame
frame = decompress_video_frame(compressed, frame_index=0, config={})
print(f"Reconstructed frame shape: {frame.shape}")
EOF
```

### Regenerate Presigned URL (if expired)

```python
import boto3

s3 = boto3.client('s3')
url = s3.generate_presigned_url(
    'get_object',
    Params={
        'Bucket': 'ai-video-codec-videos-580473065386',
        'Key': 'reconstructed/proc_exp_1760710000.mp4'
    },
    ExpiresIn=604800  # 7 days
)
print(url)
```

---

## Future Enhancements

### Phase 1 (Completed) ✅
- [x] Upload videos to S3
- [x] Save decoder code
- [x] Display on dashboard
- [x] Presigned URLs

### Phase 2 (Future)
- [ ] Video comparison view (side-by-side original vs reconstructed)
- [ ] Embedded video player in dashboard
- [ ] Video quality heatmap (show artifacts)
- [ ] Decoder version history

### Phase 3 (Future)
- [ ] On-demand video regeneration API
- [ ] Video transcoding (different formats)
- [ ] Frame-by-frame comparison tool
- [ ] Download compressed data + decoder as package

---

## Troubleshooting

### Video Not Showing

**Check 1:** Experiment achieved target?
```bash
aws dynamodb get-item \
  --table-name ai-video-codec-experiments \
  --key '{"experiment_id":{"S":"proc_exp_XXX"},"timestamp":{"N":"XXX"}}' \
  --query 'Item.experiments.S' | jq -r . | jq .
```

Look for: `"target_achieved": true`

**Check 2:** Video uploaded to S3?
```bash
aws s3 ls s3://ai-video-codec-videos-580473065386/reconstructed/proc_exp_XXX.mp4
```

**Check 3:** Presigned URL expired?
- URLs expire after 7 days
- Regenerate using boto3 (see Usage Examples)

### Decoder Not Available

**Check:** S3 bucket policy allows public read
```bash
aws s3api get-bucket-policy --bucket ai-video-codec-videos-580473065386
```

**Fix:** Add policy if missing:
```bash
aws s3api put-bucket-policy --bucket ai-video-codec-videos-580473065386 --policy '{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::ai-video-codec-videos-580473065386/decoders/*"
  }]
}'
```

---

## Summary

✅ **Deployed:** Video upload and decoder storage for successful experiments  
✅ **Dashboard:** Beautiful "View Video" and "Download Decoder" buttons  
✅ **Cost:** ~$0.14/year for 100 experiments (negligible)  
✅ **Security:** Time-limited presigned URLs  
✅ **DynamoDB:** URLs and S3 keys stored for retrieval  

**Next successful experiment (< 1 Mbps) will automatically have video and decoder available!** 🎉

