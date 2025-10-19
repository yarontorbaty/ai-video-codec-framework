#!/bin/bash
# Setup script for GPU instance with vLLM and Llama 3.1 8B

set -e

echo "============================================================"
echo "🚀 Setting up Local LLM for Codec Generation"
echo "============================================================"
echo ""

# Update system
echo "📦 Updating system packages..."
sudo yum update -y

# Install Python 3.11
echo "🐍 Installing Python 3.11..."
sudo yum install -y python3.11 python3.11-pip

# Install CUDA drivers (already on GPU instances, but ensure latest)
echo "🎮 Checking NVIDIA drivers..."
nvidia-smi || echo "⚠️  NVIDIA drivers not detected, may need manual installation"

# Install vLLM
echo "⚡ Installing vLLM..."
pip3.11 install --upgrade pip
pip3.11 install vllm==0.5.0 fastapi uvicorn

# Install model utilities
pip3.11 install huggingface_hub

# Download model (Llama 3.1 8B - fast and decent quality)
echo "📥 Downloading Llama 3.1 8B model..."
echo "This will take 5-10 minutes..."
python3.11 << 'PYTHON_EOF'
from huggingface_hub import snapshot_download
import os

# Download model
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
cache_dir = "/home/ec2-user/models"

print(f"Downloading {model_name}...")
snapshot_download(
    repo_id=model_name,
    cache_dir=cache_dir,
    local_dir=f"{cache_dir}/llama-3.1-8b-instruct",
    local_dir_use_symlinks=False
)
print("✅ Model downloaded!")
PYTHON_EOF

# Create vLLM server script
cat > /home/ec2-user/start_vllm.sh << 'EOF'
#!/bin/bash
# Start vLLM server with Llama 3.1 8B

MODEL_PATH="/home/ec2-user/models/llama-3.1-8b-instruct"
PORT=8000

echo "🚀 Starting vLLM server..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo ""

python3.11 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size 1 \
    --dtype auto \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --served-model-name llama-3.1-8b
EOF

chmod +x /home/ec2-user/start_vllm.sh

# Start vLLM server in background
echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the server, run:"
echo "  nohup /home/ec2-user/start_vllm.sh > vllm.log 2>&1 &"
echo ""
echo "To check status:"
echo "  tail -f vllm.log"
echo "  curl http://localhost:8000/v1/models"
echo ""
echo "============================================================"

