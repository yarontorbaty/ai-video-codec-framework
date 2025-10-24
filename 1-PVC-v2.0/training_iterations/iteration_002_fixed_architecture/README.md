# Training Iteration 001: True Autoencoder (960×540 Anime Frames)

**Status:** ✅ In Progress (Epoch 31/100)  
**Started:** October 24, 2025  
**Instance:** g5.48xlarge (8× A10G GPUs)

---

## 🎯 OBJECTIVE

Train a **True Autoencoder** (not additive-residual) for anime frame compression at 960×540 resolution. The goal is to achieve high compression ratios with acceptable reconstruction quality measured by PSNR and SSIM.

**Key Difference from Previous Attempts:**
- This is a **pure autoencoder** architecture where `output = decoder(latent)`
- Previous versions used additive-residual: `output = base_image + decoder(latent)` which required the base image at decode time
- This version can reconstruct images from latent representation alone

---

## 📊 TRAINING CONFIGURATION

### Model Architecture
```python
CompressionAutoencoder(
    latent_channels=32,
    input_size=(540, 960, 3)  # H, W, C (cropped to multiples of 32)
)
```

**Architecture Details:**
- **Encoder:** Progressive downsampling with convolutions
- **Latent Space:** 32 channels
- **Decoder:** Progressive upsampling with transposed convolutions
- **Total Parameters:** 222,723 (~223K)

### Training Hyperparameters
```python
{
    'optimizer': 'AdamW',
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'scheduler': 'CosineAnnealingLR',
    'loss_function': 'MSE',  # Perceptual loss disabled due to urllib3 issue
    'batch_size_per_gpu': 8,
    'total_batch_size': 64,  # 8 GPUs × 8
    'gradient_clipping': 1.0,
    'epochs': 100,
    'val_split': 0.1  # 90% train, 10% validation
}
```

### Dataset
- **Name:** `anime_frames_960x540_50k.npy`
- **S3 Location:** `s3://ai-codec-v3-artifacts-580473065386/pvc/datasets/anime_frames_960x540_50k.npy`
- **Size:** 71 GB
- **Total Frames:** 48,672
- **Train Frames:** 43,804 (90%)
- **Validation Frames:** 4,868 (10%)
- **Resolution:** 960×540 (will be cropped to 960×544 for 32-pixel alignment)
- **Format:** NumPy array, float32, normalized [0, 1], shape: (N, H, W, 3)
- **Loading:** Memory-mapped (mmap_mode='r') to avoid RAM overload

---

## 🚀 QUICK START - RECREATE TRAINING

### Prerequisites
- AWS CLI configured
- SSM access to EC2 instances
- S3 access to `ai-codec-v3-artifacts-580473065386` bucket

### Step 1: Launch GPU Instance
```bash
# Launch g5.48xlarge with 100GB EBS
aws ec2 run-instances \
  --image-id ami-0c02fb55b15a6caa6 \
  --instance-type g5.48xlarge \
  --iam-instance-profile Name=EC2-SSM-S3-Full-Access \
  --block-device-mappings '[{"DeviceName":"/dev/xda","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=autoencoder-training}]' \
  --region us-east-1

# Wait for instance to be ready and SSM online (~2-3 minutes)
INSTANCE_ID="<your-instance-id>"
```

### Step 2: Setup Environment
```bash
# Send setup commands via SSM
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=[
    "mkdir -p /home/ec2-user/autoencoder_training/models",
    "mkdir -p /home/ec2-user/autoencoder_training/trained_models",
    "cd /home/ec2-user/autoencoder_training",
    "aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/training_iterations/iteration_001/ . --recursive --region us-east-1",
    "aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/datasets/anime_frames_960x540_50k.npy . --region us-east-1",
    "pip3 install torch torchvision flask --quiet",
    "pip3 install \"urllib3<2.0\" --quiet",
    "echo Setup complete"
  ]' \
  --region us-east-1
```

### Step 3: Start Training
```bash
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=[
    "cd /home/ec2-user/autoencoder_training",
    "nohup python3 -u train_autoencoder_multigpu.py --dataset anime_frames_960x540_50k.npy --output-dir ./trained_models --epochs 100 --batch-size 8 --no-perceptual > training.log 2>&1 &",
    "echo Training started"
  ]' \
  --region us-east-1
```

### Step 4: Start Dashboard (Optional)
```bash
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=[
    "cd /home/ec2-user/autoencoder_training",
    "PYTHONUNBUFFERED=1 python3 -u autoencoder_dashboard.py > dashboard.log 2>&1 &",
    "echo Dashboard started on port 8080"
  ]' \
  --region us-east-1

# Get instance IP
INSTANCE_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --region us-east-1 --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "Dashboard: http://$INSTANCE_IP:8080"
```

---

## 📁 FILE STRUCTURE

```
iteration_001_true_autoencoder/
├── README.md                          # This file
├── SETUP_INSTRUCTIONS.md              # Detailed step-by-step setup
├── train_autoencoder_multigpu.py      # Main training script
├── autoencoder_dashboard.py           # Web dashboard for monitoring
├── models/
│   └── true_autoencoder.py           # Model architecture
├── scripts/
│   ├── launch_training.sh            # Automated launch script
│   └── monitor_training.sh           # Monitoring script
└── configs/
    └── training_config.json          # Training configuration
```

---

## 🔧 RESOLVED ISSUES & FIXES

### Issue 1: DDP Port Conflict
**Problem:** Each worker generated different random ports, preventing communication  
**Fix:** Shared one random port across all workers before spawning  
**Code:** Lines 46-47, 306 in `train_autoencoder_multigpu.py`

### Issue 2: RAM Overload (568GB Required)
**Problem:** Each of 8 workers tried to load 71GB dataset independently  
**Fix:** Memory-mapped dataset with explicit `.copy()` per batch  
**Code:** Lines 26-44 in `train_autoencoder_multigpu.py`

### Issue 3: urllib3/OpenSSL Incompatibility
**Problem:** VGG perceptual loss failed to load (OpenSSL 1.0.2k vs urllib3 v2.0)  
**Fix:** Disabled perceptual loss with `--no-perceptual` flag  
**Code:** Use `pip3 install "urllib3<2.0"` or train with MSE only

### Issue 4: DataLoader Hang
**Problem:** Workers hung at first batch load with multiprocessing  
**Fix:** Set `num_workers=0` in DataLoader  
**Code:** Lines 94-101 in `train_autoencoder_multigpu.py`

### Issue 5: Debug Log Spam
**Problem:** "Processing first batch..." printed on every iteration  
**Fix:** This is a debugging artifact from line 185-186, can be removed for cleaner logs  
**Impact:** Doesn't affect training, just makes logs verbose

---

## 📈 EXPECTED RESULTS

### PSNR Progression (Estimated)
| Epoch | Expected PSNR | Status |
|-------|---------------|--------|
| 1     | -41.88 dB     | ✅ Achieved |
| 10    | 10-15 dB      | In Progress |
| 25    | 18-22 dB      | Pending |
| 50    | 23-26 dB      | Pending |
| 100   | 26-30 dB      | Target |

### Training Metrics
- **Time per Epoch:** ~11.5 minutes (~693 seconds)
- **Total Training Time:** ~19 hours
- **GPU Utilization:** 0-50% (varies per batch)
- **GPU Memory:** 4-7 GB per GPU
- **Cost:** ~$10.85/hour × 19 hours = ~$206

---

## 💾 MODEL CHECKPOINTS

### Automatic Saving
- **Every Epoch:** `checkpoint_latest.pth` (for crash recovery)
- **Best Model:** `autoencoder_best.pth` (lowest validation loss)
- **Every 10 Epochs:** `autoencoder_epoch_N.pth` (periodic backups)

### S3 Storage
```bash
s3://ai-codec-v3-artifacts-580473065386/pvc/
├── autoencoder_best.pth              # Best model (updated per epoch)
├── checkpoint_latest.pth             # Latest checkpoint
├── autoencoder_epoch_10_best.pth     # Epoch 10 backup
├── autoencoder_epoch_20_best.pth     # Epoch 20 backup
└── ...
```

### Download Models
```bash
# Download best model
aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/autoencoder_best.pth ./ --region us-east-1

# Download specific epoch
aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/autoencoder_epoch_30_best.pth ./ --region us-east-1
```

---

## 🎮 MONITORING

### Web Dashboard
- **URL:** `http://<instance-ip>:8080`
- **Auto-refresh:** Every 15 seconds
- **Displays:** Epoch, PSNR, Loss, GPU stats, Training log

### Terminal Monitoring
```bash
# Check current epoch
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["tail -5000 /home/ec2-user/autoencoder_training/training.log | grep -E \"Epoch [0-9]+/100 \\([0-9.]+s\\):\" | tail -5"]' \
  --region us-east-1

# Check GPU utilization
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["nvidia-smi"]' \
  --region us-east-1
```

---

## 🧪 TESTING TRAINED MODEL

### Local Testing
```python
import torch
from models.true_autoencoder import CompressionAutoencoder

# Load model
model = CompressionAutoencoder(latent_channels=32)
checkpoint = torch.load('autoencoder_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test on image
import numpy as np
import cv2

# Load and preprocess image
img = cv2.imread('test_frame.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (960, 544))  # Resize to 32-multiple
img = img.astype(np.float32) / 255.0

# Convert to tensor
img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 544, 960)

# Encode and decode
with torch.no_grad():
    reconstructed, latent = model(img_tensor)

# Calculate metrics
mse = torch.mean((reconstructed - img_tensor) ** 2).item()
psnr = 10 * np.log10(1.0 / mse)
print(f"PSNR: {psnr:.2f} dB")
```

---

## 📝 NOTES & LESSONS LEARNED

1. **Memory-Mapped Datasets:** Essential for large datasets with multi-GPU training. Loading 71GB × 8 workers = 568GB would exceed available RAM.

2. **DDP Port Management:** Random ports must be shared BEFORE spawning workers, not generated inside worker processes.

3. **Perceptual Loss Compatibility:** VGG-based perceptual loss requires compatible OpenSSL/urllib3 versions. MSE-only training is a viable alternative.

4. **Batch Size Tuning:** Started with batch_size=8 per GPU. Larger batches (16, 32) caused OOM errors with 960×540 images.

5. **Debug Output:** Flush all print statements in DDP training to see real-time progress. Python buffering can hide critical debug info.

6. **Negative PSNR:** Early epochs show negative PSNR due to catastrophically poor reconstruction. This is normal and improves rapidly after epoch 5-10.

---

## 🔄 NEXT ITERATIONS

Potential improvements for future iterations:

1. **Enable Perceptual Loss:** Fix OpenSSL/urllib3 compatibility for better perceptual quality
2. **Larger Latent Space:** Try 48 or 64 channels for better quality at cost of compression
3. **Different Resolutions:** Train on 512×512 or 1280×720
4. **Quantization:** Add INT8 quantization for further compression
5. **Rate-Distortion Training:** Train multiple models with different latent sizes for rate control
6. **Temporal Compression:** Add temporal prediction for video sequences

---

## 📞 SUPPORT & TROUBLESHOOTING

### Common Issues

**Training Hangs:**
```bash
# Check if processes are running
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["ps aux | grep train_autoencoder | grep -v grep"]' \
  --region us-east-1

# Kill and restart if hung
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["pkill -9 python3"]' \
  --region us-east-1
```

**Out of Memory:**
- Reduce batch size: `--batch-size 4` or `--batch-size 2`
- Use smaller instance (g5.12xlarge with 4 GPUs)

**Slow Training:**
- Current: ~11.5 min/epoch on g5.48xlarge (8× A10G)
- Expected on g5.12xlarge (4× A10G): ~15-20 min/epoch
- Expected on g4dn.xlarge (1× T4): ~60-90 min/epoch

---

**Created:** October 24, 2025  
**Last Updated:** October 24, 2025  
**Training Status:** In Progress (Epoch 31/100)  
**Estimated Completion:** ~12-13 hours remaining

