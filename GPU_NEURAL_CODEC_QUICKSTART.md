# 🚀 GPU-First Neural Codec - Quick Start Guide

## Overview

This guide will help you deploy and run the GPU-first two-agent neural video codec system.

---

## 🎯 Prerequisites

### AWS Resources
- ✅ AWS Account with admin access
- ✅ SQS Queue: `ai-video-codec-training-queue`
- ✅ DynamoDB Table: `ai-video-codec-experiments`
- ✅ S3 Bucket: `ai-video-codec-videos-<account-id>`
- ✅ IAM Roles with appropriate permissions

### Hardware Requirements
- **Orchestrator**: t3.medium (2 vCPU, 4 GB RAM) - CPU only
- **GPU Worker**: g4dn.xlarge (4 vCPU, 16 GB RAM, NVIDIA T4 16GB)

---

## 📦 Installation

### 1. Setup Orchestrator Instance

```bash
# Launch EC2 instance
# AMI: Ubuntu 22.04 LTS
# Instance type: t3.medium
# IAM role: Allow SQS, DynamoDB, S3 access

# SSH into instance
ssh ubuntu@<orchestrator-ip>

# Clone repository
git clone https://github.com/yarontorbaty/ai-video-codec-framework.git
cd ai-video-codec-framework

# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Install optional dependencies for LLM
pip3 install anthropic openai

# Configure AWS credentials
aws configure
```

### 2. Setup GPU Worker Instance

```bash
# Launch EC2 instance
# AMI: Deep Learning AMI (Ubuntu 22.04) - has CUDA pre-installed
# Instance type: g4dn.xlarge
# IAM role: Allow SQS, DynamoDB, S3 access

# SSH into instance
ssh ubuntu@<gpu-worker-ip>

# Clone repository
git clone https://github.com/yarontorbaty/ai-video-codec-framework.git
cd ai-video-codec-framework

# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies
pip3 install scikit-image thop

# Verify GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should print: CUDA available: True

# Configure AWS credentials
aws configure
```

---

## 🚀 Running the System

### Step 1: Upload Test Video to S3

```bash
# On your local machine or orchestrator
aws s3 cp test_data/SOURCE_HD_RAW.mp4 \
  s3://ai-video-codec-videos-<account-id>/test_data/SOURCE_HD_RAW.mp4
```

### Step 2: Start GPU Worker

```bash
# On GPU worker instance
cd /home/ubuntu/ai-video-codec-framework
source venv/bin/activate

# Start worker (runs continuously)
python3 workers/neural_codec_gpu_worker.py

# You should see:
# 🚀 NEURAL CODEC GPU WORKER STARTED
#    Worker ID: ip-10-0-1-123-12345
#    GPU: NVIDIA T4
#    Device: cuda
#    Queue: https://sqs.us-east-1.amazonaws.com/.../ai-video-codec-training-queue
# 📥 Polling for neural codec experiments...
#    No jobs available, waiting...
```

### Step 3: Start Orchestrator

```bash
# On orchestrator instance (separate terminal)
cd /home/ubuntu/ai-video-codec-framework
source venv/bin/activate

# Set environment variables (optional)
export EXPERIMENT_ITERATION=1
export ANTHROPIC_API_KEY=sk-...  # If using Claude for LLM
export OPENAI_API_KEY=sk-...     # If using GPT for LLM

# Start orchestrator (runs one cycle)
python3 src/agents/gpu_first_orchestrator.py

# You should see:
# ================================================================================
# 🚀 GPU-FIRST ORCHESTRATOR
#    Iteration: 1
#    No local execution - all work dispatched to GPU workers
# ================================================================================
# 🚀 GPU-FIRST EXPERIMENT CYCLE 1
#    ID: gpu_exp_1234567890
#    Timestamp: 1234567890
# ================================================================================
#
# 📐 PHASE 1: DESIGN (Orchestrator)
#    Analyzing past experiments...
#    Found 10 recent experiments
#    Generated encoding agent: 15234 chars
#    Generated decoding agent: 12456 chars
#    Strategy: hybrid_semantic
#    ✅ Design complete
#
# 📤 PHASE 2: DISPATCH TO GPU
#    Sending experiment to GPU worker queue...
#    ✅ Dispatched to SQS
#    Message ID: abc123-def456-...
#
# ⏳ PHASE 3: WAITING FOR GPU WORKER
#    Polling for results (timeout: 1800s)...
```

### Step 4: Watch GPU Worker Process Job

```bash
# On GPU worker terminal, you'll see:
# 📥 Polling for neural codec experiments...
#
# 🎯 Received job: gpu_exp_1234567890
#
# 🔬 Executing experiment: gpu_exp_1234567890
#    Device: cuda
#   📥 Loading video from s3://ai-video-codec-videos-.../SOURCE_HD_RAW.mp4
#   ✅ Loaded 300 frames at 1080x1920
#   🗜️  Executing encoding agent...
#   🎬 Encoding video: 1x300 frames at 1080x1920
#     📊 Analyzing scene...
#     Scene: moderate_motion, Complexity: 0.65, Motion: 0.45
#     → Strategy: hybrid_semantic (default)
#     🎞️  Selected 11 I-frames from 300 total frames
#     🗜️  Compressing I-frames...
#     📝 Generating semantic description...
#     ✅ Encoding complete
#   ✅ Encoding complete
#      Bitrate: 0.8432 Mbps
#      Strategy: hybrid_semantic
#   🔄 Executing decoding agent...
#   🎬 Decoding video...
#     🎞️  Decoding 11 I-frames...
#     🎨 Generating 300 frames...
#     ✨ Enhancing temporal consistency...
#     ✅ Decoding complete
#   ✅ Decoding complete
#      Decode FPS: 28.3
#      TOPS/frame: 1.12
#   📊 Calculating quality metrics...
#   ✅ Quality metrics:
#      PSNR: 36.42 dB
#      SSIM: 0.9573
# ✅ Experiment completed in 142.3s
#    Bitrate: 0.8432 Mbps
#    PSNR: 36.42 dB
#    SSIM: 0.9573
#    TOPS: 1.12 per frame
#
# ✅ Updated experiment gpu_exp_1234567890 status to completed
# ✅ Message deleted from queue
#
# ✅ Job completed. Total processed: 1
```

### Step 5: Orchestrator Receives Results

```bash
# Back on orchestrator terminal:
#    ⏳ Still waiting... (30s / 1800s)
#    ⏳ Still waiting... (60s / 1800s)
#    ...
#    ✅ GPU execution complete
#    Bitrate: 0.8432 Mbps
#    PSNR: 36.42 dB
#
# 📊 PHASE 4: ANALYSIS
#    Evaluating results against targets...
#    📈 Metrics:
#       Bitrate: 0.8432 Mbps (target: ≤1.0 Mbps) ✅
#       PSNR: 36.42 dB (target: ≥35.0 dB) ✅
#       SSIM: 0.9573 (target: ≥0.95) ✅
#       TOPS: 1.12 (target: ≤1.33) ✅
#       Compression: 127.3x
#       Bitrate reduction: 91.6% vs HEVC
#    📝 Blog post updated with results
#
# ✅ EXPERIMENT CYCLE COMPLETE
#    Total time: 156.2s
#    Success: True
```

**🎉 SUCCESS! All targets achieved on first try!**

---

## 📊 Viewing Results

### Dashboard (Real-Time)

```bash
# On your local machine
# Download dashboard files
scp ubuntu@<orchestrator-ip>:/home/ubuntu/ai-video-codec-framework/dashboard/index.html .
scp ubuntu@<orchestrator-ip>:/home/ubuntu/ai-video-codec-framework/dashboard/styles.css .
scp ubuntu@<orchestrator-ip>:/home/ubuntu/ai-video-codec-framework/dashboard/app.js .

# Open in browser
open index.html
```

Dashboard shows:
- Real-time experiment status
- Bitrate vs. time chart
- Quality (PSNR/SSIM) vs. time chart
- Success/failure rates
- Latest experiment details

### DynamoDB (Raw Data)

```bash
# Query latest experiments
aws dynamodb scan \
  --table-name ai-video-codec-experiments \
  --limit 10 \
  --output json | jq '.Items[] | {id: .experiment_id.S, bitrate: .gpu_results.M.bitrate_mbps.N, psnr: .gpu_results.M.psnr_db.N}'
```

---

## 🔄 Running Continuous Experiments

To run experiments continuously (autonomous evolution):

### Option 1: Loop Script

```bash
# On orchestrator instance
cd /home/ubuntu/ai-video-codec-framework
source venv/bin/activate

# Run continuous loop
for i in {1..100}; do
  echo "======================================"
  echo "ITERATION $i"
  echo "======================================"
  
  export EXPERIMENT_ITERATION=$i
  python3 src/agents/gpu_first_orchestrator.py
  
  # Wait 60 seconds between experiments
  echo "Waiting 60s before next iteration..."
  sleep 60
done
```

### Option 2: Cron Job

```bash
# Edit crontab
crontab -e

# Add entry to run every 5 minutes
*/5 * * * * cd /home/ubuntu/ai-video-codec-framework && source venv/bin/activate && python3 src/agents/gpu_first_orchestrator.py >> /tmp/orchestrator.log 2>&1
```

### Option 3: Systemd Service

```bash
# Create service file
sudo nano /etc/systemd/system/neural-codec-orchestrator.service

# Add content:
[Unit]
Description=Neural Codec Orchestrator
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-video-codec-framework
Environment="PATH=/home/ubuntu/ai-video-codec-framework/venv/bin:/usr/bin"
ExecStart=/home/ubuntu/ai-video-codec-framework/venv/bin/python3 src/agents/gpu_first_orchestrator.py
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable neural-codec-orchestrator
sudo systemctl start neural-codec-orchestrator

# Check status
sudo systemctl status neural-codec-orchestrator
```

---

## 🐛 Common Issues & Solutions

### Issue 1: "No GPU detected"

**Symptom**: Worker prints "No GPU detected - will use CPU"

**Solution**:
```bash
# Check CUDA installation
nvidia-smi

# If nvidia-smi fails, install NVIDIA drivers
sudo apt update
sudo apt install -y nvidia-driver-525

# Reboot
sudo reboot

# Verify PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Issue 2: "Cannot connect to SQS"

**Symptom**: Worker cannot poll SQS queue

**Solution**:
```bash
# Check IAM role attached to EC2 instance
aws sts get-caller-identity

# Verify SQS permissions
aws sqs get-queue-attributes \
  --queue-url https://sqs.us-east-1.amazonaws.com/.../ai-video-codec-training-queue \
  --attribute-names QueueArn

# If fails, attach IAM role with SQS permissions to EC2 instance
```

### Issue 3: "GPU execution timeout"

**Symptom**: Orchestrator waits 30 minutes and times out

**Solution**:
1. Check GPU worker is running: `ps aux | grep neural_codec_gpu_worker`
2. Check GPU worker logs for errors
3. Verify GPU worker can access S3: `aws s3 ls s3://ai-video-codec-videos-.../`
4. Increase timeout in orchestrator: Edit `self.gpu_timeout_seconds = 3600` (1 hour)

### Issue 4: "Out of GPU memory"

**Symptom**: Worker crashes with CUDA OOM error

**Solution**:
```bash
# Reduce batch size in config
# Edit experiment config to use smaller resolution or fewer frames

# Or upgrade to larger GPU instance (g4dn.2xlarge with 32GB)
```

### Issue 5: "LLM API key not set"

**Symptom**: Orchestrator fails in design phase

**Solution**:
```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...

# Or create .env file
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

---

## 📈 Performance Expectations

### First Experiment (Baseline)
- **Bitrate**: 2-5 Mbps (needs improvement)
- **PSNR**: 35-40 dB (good)
- **SSIM**: 0.93-0.97 (good)
- **Time**: 2-5 minutes

### After 10 Experiments
- **Bitrate**: 1-2 Mbps (getting better)
- **PSNR**: 36-40 dB (improved)
- **SSIM**: 0.95-0.98 (excellent)

### After 50 Experiments
- **Bitrate**: 0.5-1.0 Mbps ✅ (target achieved!)
- **PSNR**: 35-38 dB ✅
- **SSIM**: 0.95-0.97 ✅
- **TOPS**: <1.3 ✅

---

## 💰 Cost Estimate

### AWS Costs (On-Demand, us-east-1)

**Orchestrator** (t3.medium, always on):
- Instance: $0.0416/hour × 730 hours/month = **$30.37/month**

**GPU Worker** (g4dn.xlarge, 4 hours/day):
- Instance: $0.526/hour × 4 hours/day × 30 days = **$63.12/month**

**Storage**:
- S3: 100 GB × $0.023/GB = **$2.30/month**
- DynamoDB: Free tier (25 GB) = **$0/month**
- SQS: Free tier (1M requests) = **$0/month**

**Total**: ~**$95-100/month**

### Cost Optimization

**Use Spot Instances for GPU Workers**:
- Spot price: ~$0.16/hour (70% cheaper)
- Monthly: $0.16 × 120 hours = **$19.20/month**
- **Total with spot: ~$52/month**

**Use Lambda for Orchestrator** (future):
- Lambda: $0.20 per 1M requests
- Monthly: negligible
- **Total: ~$20/month**

---

## 🎯 Next Steps

### 1. Monitor First 10 Experiments
Watch the system learn and improve:
- Are bitrates decreasing?
- Is quality stable or improving?
- Which strategies work best?

### 2. Tune Configuration
Edit `config/ai_codec_config.yaml`:
- Adjust `target_bitrate_mbps`
- Change `i_frame_interval`
- Modify `latent_dim`, `description_dim`

### 3. Try Different Videos
Upload various content types:
- Static scenes (security camera)
- Talking heads (video calls)
- High motion (sports, action)
- See which strategies work best for each

### 4. Scale Up
Add more GPU workers:
- Parallel experiment execution
- Faster iteration cycles
- More diverse approaches

### 5. Deploy to Edge
Once decoder is optimized:
- Export to ONNX
- Quantize to INT8
- Deploy to mobile app
- Test real-time performance

---

## 📚 Additional Resources

- **Architecture Doc**: `GPU_NEURAL_CODEC_ARCHITECTURE.md`
- **LLM Prompt**: `LLM_SYSTEM_PROMPT_V2.md`
- **Code**:
  - Encoding: `src/agents/encoding_agent.py`
  - Decoding: `src/agents/decoding_agent.py`
  - Orchestrator: `src/agents/gpu_first_orchestrator.py`
  - Worker: `workers/neural_codec_gpu_worker.py`

---

## ✅ Checklist

Before running your first experiment, make sure:

- [ ] AWS account configured with access keys
- [ ] SQS queue created: `ai-video-codec-training-queue`
- [ ] DynamoDB table created: `ai-video-codec-experiments`
- [ ] S3 bucket created and test video uploaded
- [ ] Orchestrator EC2 instance launched and code installed
- [ ] GPU worker EC2 instance launched with CUDA support
- [ ] PyTorch with CUDA installed on GPU worker
- [ ] Both instances have IAM roles with SQS/DynamoDB/S3 permissions
- [ ] GPU worker is running and polling queue
- [ ] LLM API key set (Anthropic or OpenAI)

---

## 🚀 Ready to Launch!

```bash
# Start GPU worker (Terminal 1)
ssh ubuntu@<gpu-worker-ip>
cd ai-video-codec-framework && source venv/bin/activate
python3 workers/neural_codec_gpu_worker.py

# Start orchestrator (Terminal 2)
ssh ubuntu@<orchestrator-ip>
cd ai-video-codec-framework && source venv/bin/activate
python3 src/agents/gpu_first_orchestrator.py

# Watch the magic happen! 🎉
```

**Your autonomous neural codec system is now evolving toward 90% bitrate reduction!**

---

**Questions?** Check `GPU_NEURAL_CODEC_ARCHITECTURE.md` for detailed technical information.

**Issues?** See troubleshooting section above or check CloudWatch logs.

**Success?** 🎉 You're running a state-of-the-art autonomous neural video codec!

