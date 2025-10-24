# 🚀 QUICK START - Recreate Training in 3 Commands

This is the fastest way to recreate this exact training iteration.

---

## ✨ Super Quick (If you have an existing instance)

```bash
INSTANCE_ID="i-0439bc256e672279f"  # Replace with your instance ID

# Download and run the automated script
cd /Users/yarontorbaty/Documents/Code/AiV1/1-PVC-v2.0/training_iterations/iteration_001_true_autoencoder
./launch_training.sh $INSTANCE_ID
```

**Done!** The script will:
1. ✅ Setup environment
2. ✅ Download dataset (71GB)
3. ✅ Start training
4. ✅ Start dashboard
5. ✅ Give you the dashboard URL

---

## 🔥 From Scratch (No existing instance)

### Step 1: Launch Instance
```bash
aws ec2 run-instances \
  --image-id ami-0c02fb55b15a6caa6 \
  --instance-type g5.48xlarge \
  --iam-instance-profile Name=EC2-SSM-S3-Full-Access \
  --block-device-mappings '[{"DeviceName":"/dev/xda","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=autoencoder-training}]' \
  --region us-east-1

# Get the instance ID from the output
INSTANCE_ID="i-xxxxxxxxxxxxxxxxx"
```

### Step 2: Wait for SSM (~2 minutes)
```bash
# Check SSM status
aws ssm describe-instance-information \
  --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
  --region us-east-1 \
  --query 'InstanceInformationList[0].PingStatus'

# Wait for "Online"
```

### Step 3: Run Automated Setup
```bash
cd /Users/yarontorbaty/Documents/Code/AiV1/1-PVC-v2.0/training_iterations/iteration_001_true_autoencoder
./launch_training.sh $INSTANCE_ID
```

---

## 📊 Monitor Training

### Web Dashboard (Recommended)
```bash
# Get instance IP
INSTANCE_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --region us-east-1 \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "Dashboard: http://$INSTANCE_IP:8080"
# Open in browser - auto-refreshes every 15 seconds
```

### Terminal Check
```bash
# Check current epoch
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["tail -5000 /home/ec2-user/autoencoder_training/training.log | grep -E \"Epoch [0-9]+/100 \\([0-9.]+s\\):\" | tail -3"]' \
  --region us-east-1

# Get command ID, then:
aws ssm get-command-invocation \
  --command-id <command-id> \
  --instance-id $INSTANCE_ID \
  --region us-east-1 \
  --query 'StandardOutputContent' \
  --output text
```

---

## 💾 Download Models

### After Training Completes
```bash
# Download best model
aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/autoencoder_best.pth ./ --region us-east-1

# Download specific epoch
aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/autoencoder_epoch_50_best.pth ./ --region us-east-1

# Download final checkpoint
aws s3 cp s3://ai-codec-v3-artifacts-580473065386/pvc/checkpoint_latest.pth ./ --region us-east-1
```

---

## 🛑 Stop/Cleanup

### Stop Training (Keep Instance)
```bash
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["pkill -9 python3"]' \
  --region us-east-1
```

### Terminate Instance
```bash
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
```

---

## 📋 Training Details

- **Dataset:** 71GB, 48,672 anime frames at 960×540
- **Model:** 223K parameters, 32 latent channels
- **Time:** ~19 hours (100 epochs × 11.5 min)
- **Cost:** ~$206 ($10.85/hour × 19 hours)
- **Expected PSNR:** 26-30 dB at epoch 100

---

## 🔧 Troubleshooting

**Training not starting:**
```bash
# Check if process is running
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["ps aux | grep train_autoencoder | grep -v grep"]' \
  --region us-east-1
```

**Dashboard not loading:**
```bash
# Restart dashboard
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["cd /home/ec2-user/autoencoder_training && pkill -f autoencoder_dashboard.py && PYTHONUNBUFFERED=1 python3 -u autoencoder_dashboard.py > dashboard.log 2>&1 &"]' \
  --region us-east-1
```

**Out of memory:**
```bash
# Restart with smaller batch size
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["cd /home/ec2-user/autoencoder_training && pkill -9 python3 && nohup python3 -u train_autoencoder_multigpu.py --dataset anime_frames_960x540_50k.npy --output-dir ./trained_models --epochs 100 --batch-size 4 --no-perceptual > training.log 2>&1 &"]' \
  --region us-east-1
```

---

## 📂 File Locations

**Local:**
- Training package: `/Users/yarontorbaty/Documents/Code/AiV1/1-PVC-v2.0/training_iterations/iteration_001_true_autoencoder/`

**S3:**
- Training files: `s3://ai-codec-v3-artifacts-580473065386/pvc/training_iterations/iteration_001/`
- Dataset: `s3://ai-codec-v3-artifacts-580473065386/pvc/datasets/anime_frames_960x540_50k.npy`
- Models: `s3://ai-codec-v3-artifacts-580473065386/pvc/autoencoder_*.pth`

---

**That's it!** The entire training can be recreated from scratch in ~30 minutes (most of that is waiting for the 71GB dataset download).

