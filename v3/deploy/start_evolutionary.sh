#!/bin/bash

echo "🧬 Starting Evolutionary Orchestrator..."

# Stop any running orchestrators
pkill -f orchestrator.py

# Create directory
mkdir -p /home/ec2-user/evolutionary-orchestrator
cd /home/ec2-user/evolutionary-orchestrator

# Download the orchestrator
curl -s https://ai-codec-v3-artifacts-580473065386.s3.amazonaws.com/code/evolutionary_orchestrator.py -o evolutionary_orchestrator.py

# Start evolutionary orchestrator
export WORKER_URL=http://172.31.65.58:8080
nohup python3 evolutionary_orchestrator.py > evolutionary.log 2>&1 &

echo "✅ Evolutionary orchestrator started"
echo "📊 Monitoring logs..."
sleep 3
tail -30 evolutionary.log

