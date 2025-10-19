#!/bin/bash
set -e

echo "🐳 Installing Docker and NVIDIA Container Toolkit..."

# Update and install Docker
sudo apt-get update -qq
sudo apt-get install -y docker.io

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update -qq
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "✅ Docker and NVIDIA toolkit installed!"

echo "🚀 Starting vLLM container with Llama 3.1 8B..."

# Pull and run vLLM with Llama 3.1 8B
sudo docker run -d \
  --gpus all \
  --name vllm \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dtype auto \
  --max-model-len 4096

echo "✅ vLLM container started!"
echo "📊 Checking status..."
sleep 10
sudo docker logs vllm --tail 50

echo "✅ Setup complete! vLLM is running on port 8000"

