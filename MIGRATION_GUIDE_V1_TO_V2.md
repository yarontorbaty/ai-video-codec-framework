# 🔄 Migration Guide: v1.0 → v2.0 GPU-First Neural Codec

**Date**: October 17, 2025

---

## 📋 Overview

This guide helps you transition from the v1.0 system (local execution) to v2.0 (GPU-first neural codec).

---

## 🔍 What Changed

### Architecture

**v1.0**:
```
Orchestrator (CPU)
  ↓
Run compression locally
  ↓
Measure quality
  ↓
Store results
```

**v2.0**:
```
Orchestrator (CPU - Coordinator only)
  ↓
Dispatch to GPU Worker (SQS)
  ↓
GPU Worker:
  • EncodingAgent (compress)
  • DecodingAgent (reconstruct)
  • Quality measurement
  ↓
Store results (DynamoDB)
  ↓
Orchestrator analyzes
```

---

## 🗂️ File Mapping

### Deprecated Files (v1.0)

| Old File | Status | Replacement |
|----------|--------|-------------|
| `src/agents/procedural_experiment_runner.py` | ⚠️ DEPRECATED | `src/agents/gpu_first_orchestrator.py` |
| `workers/training_worker.py` | ⚠️ DEPRECATED | `workers/neural_codec_gpu_worker.py` |
| `LLM_SYSTEM_PROMPT.md` | ⚠️ DEPRECATED | `LLM_SYSTEM_PROMPT_V2.md` |

**Note**: Old files are NOT deleted (for reference) but should not be used.

### New Files (v2.0)

| File | Purpose |
|------|---------|
| `src/agents/encoding_agent.py` | Encoding neural network agent |
| `src/agents/decoding_agent.py` | Decoding neural network agent (40 TOPS) |
| `src/agents/gpu_first_orchestrator.py` | GPU-first orchestrator |
| `workers/neural_codec_gpu_worker.py` | Neural codec GPU worker |
| `LLM_SYSTEM_PROMPT_V2.md` | Updated LLM instructions |
| `GPU_NEURAL_CODEC_ARCHITECTURE.md` | Technical architecture |
| `GPU_NEURAL_CODEC_QUICKSTART.md` | Setup guide |
| `GPU_NEURAL_CODEC_IMPLEMENTATION_COMPLETE.md` | Implementation summary |
| `V2_NEURAL_CODEC_README.md` | Project README |
| `IMPLEMENTATION_SUMMARY.md` | Detailed summary |
| `MIGRATION_GUIDE_V1_TO_V2.md` | This file |

---

## 🚀 Migration Steps

### Step 1: Review v2.0 Architecture

**Read these files** (15-30 minutes):
1. `IMPLEMENTATION_SUMMARY.md` - What changed and why
2. `GPU_NEURAL_CODEC_QUICKSTART.md` - How to run v2.0
3. `LLM_SYSTEM_PROMPT_V2.md` - New LLM instructions

### Step 2: Update Infrastructure (If Needed)

**Orchestrator**:
- Instance type: t3.medium (same as v1.0) ✅
- No changes needed, but code must be updated

**GPU Worker**:
- Instance type: g4dn.xlarge (NEW - requires GPU)
- Needs CUDA and PyTorch with GPU support

**AWS Resources**:
- SQS Queue: Already exists ✅
- DynamoDB: Already exists ✅
- S3: Already exists ✅

### Step 3: Stop v1.0 Services

```bash
# On orchestrator instance
# Stop old runner if running
pkill -f procedural_experiment_runner.py

# Or if using systemd
sudo systemctl stop neural-codec-orchestrator
```

### Step 4: Update Code

```bash
# On orchestrator instance
cd /home/ubuntu/ai-video-codec-framework
git pull origin main

# Verify new files exist
ls src/agents/encoding_agent.py
ls src/agents/decoding_agent.py
ls src/agents/gpu_first_orchestrator.py
ls workers/neural_codec_gpu_worker.py
```

### Step 5: Setup GPU Worker (NEW)

```bash
# Launch new EC2 instance
# - AMI: Deep Learning AMI (Ubuntu 22.04)
# - Instance type: g4dn.xlarge
# - IAM role: Same as orchestrator (SQS, DynamoDB, S3 access)

# SSH into GPU worker
ssh ubuntu@<gpu-worker-ip>

# Clone repository
git clone https://github.com/yarontorbaty/ai-video-codec-framework.git
cd ai-video-codec-framework

# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Install PyTorch with CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies
pip3 install scikit-image thop

# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Should print: CUDA: True

# Configure AWS
aws configure
```

### Step 6: Start v2.0 Services

```bash
# Terminal 1: Start GPU worker
ssh ubuntu@<gpu-worker-ip>
cd ai-video-codec-framework
source venv/bin/activate
python3 workers/neural_codec_gpu_worker.py

# Terminal 2: Start orchestrator
ssh ubuntu@<orchestrator-ip>
cd ai-video-codec-framework
source venv/bin/activate
export ANTHROPIC_API_KEY=sk-ant-...  # Or OpenAI key
python3 src/agents/gpu_first_orchestrator.py
```

### Step 7: Verify Migration

**Check GPU Worker logs**:
```
🚀 NEURAL CODEC GPU WORKER STARTED
   Worker ID: ip-10-0-1-123-12345
   GPU: NVIDIA T4
   Device: cuda
📥 Polling for neural codec experiments...
```

**Check Orchestrator logs**:
```
🚀 GPU-FIRST ORCHESTRATOR
   Iteration: 1
   No local execution - all work dispatched to GPU workers
📐 PHASE 1: DESIGN (Orchestrator)
   Analyzing past experiments...
```

**Watch for first experiment**:
- GPU worker receives job
- Executes encoding + decoding
- Uploads results
- Orchestrator receives results and analyzes

**Success indicators**:
- ✅ Orchestrator dispatches to SQS
- ✅ GPU worker receives and processes
- ✅ Results appear in DynamoDB
- ✅ No errors in logs

---

## 🔧 Configuration Changes

### Environment Variables

**v1.0**:
```bash
export EXPERIMENT_ITERATION=1
```

**v2.0** (same, plus optional):
```bash
export EXPERIMENT_ITERATION=1
export ANTHROPIC_API_KEY=sk-ant-...  # For LLM
export OPENAI_API_KEY=sk-...         # Alternative LLM
```

### DynamoDB Schema

**No migration needed** - v2.0 uses same table with additional fields:

**New fields** (backward compatible):
- `gpu_status`: "processing", "completed", "failed"
- `gpu_results`: Metrics from GPU worker
- `gpu_worker_id`: Which worker processed
- `gpu_completed_at`: Timestamp

**Existing data**: Not affected, can coexist

---

## 📊 Performance Comparison

| Aspect | v1.0 | v2.0 |
|--------|------|------|
| **Execution Location** | Orchestrator (CPU) | GPU Workers |
| **Execution Speed** | Slow (CPU) | 10-100x faster (GPU) |
| **Codec Type** | Single agent | Two agents (encoding + decoding) |
| **Compression** | Fixed strategies | Scene-adaptive |
| **Traditional Codecs** | Not integrated | x264/265/AV1/VVC support |
| **Edge Deployment** | Not optimized | 40 TOPS constraint |
| **Scalability** | Single instance | Multiple GPU workers |
| **Cost** | $30/month | $55-100/month (with spot: $50/month) |

---

## ⚠️ Breaking Changes

### 1. Orchestrator No Longer Executes Locally

**Impact**: If you have code that assumes local execution, it will break.

**Fix**: Use GPU-first orchestrator that dispatches to workers.

### 2. Different Function Signatures

**v1.0** (old):
```python
def compress_video_frame(frame: np.ndarray, frame_index: int, config: dict) -> bytes:
    pass
```

**v2.0** (new):
```python
def compress_video_tensor(frames: torch.Tensor, config: Dict, device: str) -> Dict:
    # Returns: {compressed_data, bitrate_mbps, compression_ratio, ...}
    pass

def decompress_video_tensor(compressed_data: Dict, config: Dict, device: str) -> Dict:
    # Returns: {reconstructed_frames, decode_fps, tops_per_frame, ...}
    pass
```

**Impact**: Old compression functions won't work.

**Fix**: LLM generates new code in v2.0 format automatically.

### 3. Two Agents Required

**Impact**: Must generate both encoding and decoding agents.

**Fix**: LLM prompt (v2.0) generates both automatically.

---

## 🔄 Backward Compatibility

### What Still Works

✅ **DynamoDB Table**: Same table, new fields added
✅ **S3 Bucket**: Same bucket for videos
✅ **SQS Queue**: Same queue, new message format
✅ **Dashboard**: Still works, shows new experiments
✅ **Experiment IDs**: Still use `experiment_id` + `timestamp`

### What Doesn't Work

❌ **Old orchestrator** with new GPU worker (incompatible)
❌ **New orchestrator** with old worker (incompatible)
❌ **Old function signatures** (single-agent codec)

**Solution**: Use v2.0 components together (orchestrator + GPU worker + new agents)

---

## 📈 Rollback Plan

If v2.0 doesn't work, you can roll back:

### Step 1: Stop v2.0 Services

```bash
# Stop GPU worker
pkill -f neural_codec_gpu_worker.py

# Stop new orchestrator
pkill -f gpu_first_orchestrator.py
```

### Step 2: Restart v1.0 Services

```bash
# On orchestrator
cd ai-video-codec-framework
source venv/bin/activate
python3 src/agents/procedural_experiment_runner.py
```

### Step 3: Optionally Terminate GPU Worker

```bash
# From AWS console or CLI
aws ec2 terminate-instances --instance-ids i-xxxxx
```

**Note**: v1.0 and v2.0 data in DynamoDB are separate (different experiment IDs), so no data conflicts.

---

## 🐛 Troubleshooting Migration Issues

### Issue 1: "No GPU detected" on GPU Worker

**Cause**: CUDA not installed or PyTorch CPU-only version

**Fix**:
```bash
# Check GPU
nvidia-smi

# Reinstall PyTorch with CUDA
pip3 uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Issue 2: Orchestrator Timeout Waiting for GPU

**Cause**: GPU worker not running or not polling SQS

**Fix**:
```bash
# Check GPU worker is running
ssh ubuntu@<gpu-worker-ip>
ps aux | grep neural_codec_gpu_worker

# Check SQS queue has messages
aws sqs get-queue-attributes \
  --queue-url https://sqs.us-east-1.amazonaws.com/.../ai-video-codec-training-queue \
  --attribute-names ApproximateNumberOfMessages

# Restart GPU worker if needed
python3 workers/neural_codec_gpu_worker.py
```

### Issue 3: LLM Not Generating Code

**Cause**: API key not set

**Fix**:
```bash
# Set key
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...

# Verify
echo $ANTHROPIC_API_KEY
```

### Issue 4: "Cannot import torch" on GPU Worker

**Cause**: Virtual environment not activated or PyTorch not installed

**Fix**:
```bash
# Activate venv
cd ai-video-codec-framework
source venv/bin/activate

# Install PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## ✅ Migration Checklist

Before going live with v2.0:

- [ ] Read implementation summary and architecture docs
- [ ] Understand two-agent approach
- [ ] Setup GPU worker EC2 instance
- [ ] Install CUDA and PyTorch on GPU worker
- [ ] Configure AWS credentials on both instances
- [ ] Upload test video to S3
- [ ] Set LLM API key on orchestrator
- [ ] Stop v1.0 services
- [ ] Start v2.0 GPU worker
- [ ] Start v2.0 orchestrator
- [ ] Verify first experiment completes
- [ ] Monitor for errors in CloudWatch
- [ ] Test rollback plan (optional)

---

## 🎯 What You Gain in v2.0

1. **10-100x faster execution** (GPU vs CPU)
2. **Two-agent codec** (encoder + decoder)
3. **Scene-adaptive compression** (intelligent strategy selection)
4. **Traditional codec support** (x264/265/AV1/VVC per scene)
5. **Edge deployment ready** (40 TOPS decoder)
6. **Scalable architecture** (add more GPU workers)
7. **Semantic video generation** (GenAI-based reconstruction)
8. **Better quality/bitrate tradeoffs** (adaptive strategies)

---

## 📚 Next Steps After Migration

### 1. Run 10 Experiments

Let the system learn:
- Iteration 1-5: Baseline and exploration
- Iteration 5-10: Convergence toward targets
- Iteration 10+: Fine-tuning

### 2. Analyze Results

Check DynamoDB:
- Which strategies work best?
- Which scenes compress well?
- Is quality stable?
- Is bitrate decreasing?

### 3. Scale Up

Add more GPU workers:
- Parallel experimentation
- Faster iteration
- More diverse approaches

### 4. Optimize Costs

Use spot instances:
- 70% cost reduction
- Same performance
- Auto-restart on interruption

### 5. Deploy to Edge

Once targets met:
- Export decoder to ONNX
- Quantize to INT8
- Test on mobile devices
- Build demo app

---

## 🎉 Conclusion

**v2.0 is a major upgrade** with GPU-first architecture, two-agent codec, and scene-adaptive compression.

Migration is straightforward:
1. Setup GPU worker
2. Stop v1.0 services
3. Start v2.0 services
4. Watch autonomous evolution

**Estimated migration time**: 1-2 hours

**Questions?** See documentation files or troubleshooting section above.

**Ready to migrate?** Follow the checklist above!

---

**Welcome to v2.0!** 🚀

GPU-first, two-agent, scene-adaptive neural video codec with 40 TOPS edge deployment.

---

**Date**: October 17, 2025
**Version**: v2.0
**Status**: ✅ Ready for Migration


