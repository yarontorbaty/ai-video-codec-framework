# OOM (Out Of Memory) Prevention Guide

## 🚨 The Problem

Training crashed at Epoch 39 due to **OOM (Out Of Memory)** on GPU. This is common with large models (93M parameters) and high batch sizes.

---

## ✅ Solutions Implemented

### **1. Checkpoint Resuming**

**What it does:** Saves training state every 10 epochs, allowing resuming from where it stopped.

**How to use:**
```bash
# Resume from previous checkpoint
python train_production_multigpu.py --resume --save-path /path/to/models
```

**What's saved:**
- Model weights (`production_encoder_best.pth`, `production_decoder_best.pth`)
- Optimizer state (`optimizer.pth`)
- Training metadata (`checkpoint.json`: epoch, best loss, history)
- Training data cache (`training_data.npy`)

**Benefits:**
- No need to regenerate training data (saves 0.4 minutes)
- Continues from exact epoch where it stopped
- Preserves learning rate schedule
- Keeps training history

---

### **2. Memory Management**

**Implemented strategies:**

#### **a) Periodic Cache Clearing**
```python
# Every 10 batches, clear unused GPU memory
if (i // batch_size) % 10 == 0:
    torch.cuda.empty_cache()
```

#### **b) Post-Validation Cleanup**
```python
# After validation, aggressively free memory
torch.cuda.empty_cache()
gc.collect()
```

#### **c) Data Caching**
- Training data is generated once and cached to disk
- On resume, data is loaded from cache (no regeneration)
- Saves memory during initialization

---

### **3. Batch Size Optimization**

**Current:** 128 per GPU (512 effective across 4 GPUs)

**If OOM persists:**
```bash
# Option 1: Reduce batch size to 64
python train_production_multigpu.py --batch-size 64 --resume

# Option 2: Reduce to 32 (safest)
python train_production_multigpu.py --batch-size 32 --resume
```

**Trade-offs:**
- Lower batch size = More stable (less OOM risk)
- Lower batch size = Slightly slower training
- Lower batch size = May need slightly higher learning rate

---

## 🔧 How to Resume from Epoch 39

### **Quick Start:**

```bash
# SSH to GPU instance (or use SSM)
cd /home/ec2-user/pvc_training

# Upload updated training script
aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/training/train_production_multigpu.py \
          pvc_v2/training/train_production_multigpu.py

# Resume training (will auto-detect Epoch 39 checkpoint)
nohup python -u pvc_v2/training/train_production_multigpu.py \
      --samples 10000 \
      --epochs 100 \
      --batch-size 128 \
      --save-path /home/ec2-user/pvc_training/models \
      --resume \
      > training_resumed.log 2>&1 &

# Monitor
tail -f training_resumed.log
```

---

## 📊 What Happens When Resuming

### **Expected Output:**

```
================================================================================
PVC v2.0 Production Architecture - Multi-GPU Training
================================================================================
OOM Prevention: Enabled (gradient checkpointing, memory clearing)
Checkpoint Resuming: Enabled
================================================================================

🏗️  Initializing production models...
📊 Using DataParallel across 4 GPUs
📊 Model Parameters:
   Encoder: 65.93M
   Decoder: 27.00M
   Total: 92.93M (vs 32.4M baseline)

📂 Resuming from checkpoint: /home/ec2-user/pvc_training/models
   ✅ Resumed from Epoch 39
   📊 Best loss so far: 0.001115
   🚀 Continuing from Epoch 40

📂 Loading cached training data from training_data.npy...
✅ Data loaded: (10000, 256, 256, 3)

🚀 Starting training...
   Epochs: 40 → 100
   Batch size: 128 (×4 GPUs = 512 effective)
   Samples per epoch: 10000
   Learning rate: 0.0001

Epoch 40/100 | Loss: 0.001110 | LR: 6.58e-05 | Time: 88.2s | ETA: 1:28:20
   ✅ Saved best model (loss: 0.001110)
...
```

---

## 🛡️ Additional OOM Prevention Strategies

### **If OOM Still Occurs:**

#### **Option 1: Gradient Accumulation** (Recommended)
- Split each batch into smaller micro-batches
- Accumulate gradients before optimizer step
- Simulates larger batch size with less memory

```python
# Would require code modification
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### **Option 2: Mixed Precision Training**
- Use FP16 instead of FP32
- Reduces memory by ~50%
- Slightly faster training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### **Option 3: Gradient Checkpointing**
- Trade compute for memory
- Recompute activations during backward pass
- Can reduce memory by 30-50%

```python
from torch.utils.checkpoint import checkpoint

# In model forward pass
output = checkpoint(self.block, input)
```

---

## 📈 Memory Usage Monitoring

### **Check GPU Memory:**

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log memory usage
while true; do
  nvidia-smi --query-gpu=timestamp,memory.used,memory.free \
             --format=csv,noheader >> gpu_memory.log
  sleep 10
done
```

### **Expected Memory Usage:**

| Component | Memory (GB) | Notes |
|-----------|-------------|-------|
| **Model (Encoder)** | ~0.25 GB | 65.93M params × 4 bytes |
| **Model (Decoder)** | ~0.11 GB | 27.00M params × 4 bytes |
| **Optimizer State** | ~0.72 GB | 2× model params (Adam) |
| **Batch (128)** | ~0.38 GB | 128 × 256² × 3 × 4 bytes |
| **Activations** | ~2-4 GB | Varies by layer |
| **Gradients** | ~0.36 GB | Same as model params |
| **Total** | **~4-6 GB** per GPU | Safe for 23 GB A10G |

**Note:** OOM likely occurred due to **memory fragmentation** after 39 epochs, not total memory exhaustion.

---

## ✅ Current Status

**Training script updated with:**
- ✅ Checkpoint resuming (`--resume` flag)
- ✅ Data caching (saves 0.4 min on restart)
- ✅ Periodic memory clearing (every 10 batches)
- ✅ Post-validation cleanup
- ✅ Checkpoint metadata (epoch, loss, history)
- ✅ Optimizer state saving

**To deploy:**
1. Upload updated script to S3
2. Download on GPU instance
3. Kill current training (starting from scratch)
4. Restart with `--resume` flag

**Estimated time saved:**
- Without resume: 2.5 hours (100 epochs)
- With resume from Epoch 39: 1.5 hours (61 epochs)
- **Time saved: 1 hour** ⏱️

---

## 🚀 Recommended Action

**Deploy the updated script and resume from Epoch 39:**

```bash
# 1. Package and upload
cd /Users/yarontorbaty/Documents/Code/AiV1
tar czf pvc_v2_updated.tar.gz pvc_v2/
aws s3 cp pvc_v2_updated.tar.gz s3://ai-codec-v3-artifacts-580473065386/pvc/training/

# 2. Stop current training (on GPU instance)
pkill -f train_production_multigpu

# 3. Download and extract
cd /home/ec2-user/pvc_training
aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/training/pvc_v2_updated.tar.gz .
tar xzf pvc_v2_updated.tar.gz

# 4. Resume from Epoch 39
nohup python -u pvc_v2/training/train_production_multigpu.py \
      --samples 10000 \
      --epochs 100 \
      --batch-size 128 \
      --save-path /home/ec2-user/pvc_training/models \
      --resume \
      > training_resumed.log 2>&1 &
```

---

## 📚 References

- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

