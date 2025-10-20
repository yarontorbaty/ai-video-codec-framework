#!/usr/bin/env python3
"""Deploy vLLM to Deep Learning AMI instance"""

import boto3
import time

# Configuration
INSTANCE_ID = 'i-05d394a8827bd913b'
REGION = 'us-east-1'
GPU_IP = '172.31.74.148'
WORKER_IP = '172.31.65.58'

# Initialize SSM client
ssm = boto3.client('ssm', region_name=REGION)

print("🚀 Deploying vLLM on Deep Learning AMI...")

# Setup script
setup_commands = """#!/bin/bash
set -e

echo "📦 Installing vLLM..."
pip install vllm --user --quiet

echo "🚀 Starting vLLM server..."
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dtype auto \
  --max-model-len 4096 \
  --port 8000 \
  > /home/ec2-user/vllm.log 2>&1 &

echo "✅ vLLM server started"
sleep 10

echo "📊 Process status:"
ps aux | grep vllm | grep -v grep || echo "Process not found"

echo "📋 Log tail:"
tail -30 /home/ec2-user/vllm.log
"""

# Send command
print("📤 Sending setup command...")
response = ssm.send_command(
    InstanceIds=[INSTANCE_ID],
    DocumentName='AWS-RunShellScript',
    Parameters={
        'commands': [
            'su - ec2-user -c "' + setup_commands.replace('"', '\\"').replace('\n', '; ') + '"'
        ],
        'executionTimeout': ['600']
    }
)

command_id = response['Command']['CommandId']
print(f"📋 Command ID: {command_id}")
print("⏳ Waiting for vLLM to start (this takes 2-3 minutes for model download)...")

# Wait for command
time.sleep(180)

# Get output
result = ssm.get_command_invocation(
    CommandId=command_id,
    InstanceId=INSTANCE_ID
)

print(f"\n✅ Status: {result['Status']}")
print(f"\n📊 Output:\n{result['StandardOutputContent'][-2000:]}")

if result.get('StandardErrorContent'):
    print(f"\n⚠️ Errors:\n{result['StandardErrorContent'][-1000:]}")

# Check if vLLM is running
print("\n🔍 Final status check...")
check_response = ssm.send_command(
    InstanceIds=[INSTANCE_ID],
    DocumentName='AWS-RunShellScript',
    Parameters={'commands': [
        'su - ec2-user -c "ps aux | grep vllm | grep -v grep"',
        'su - ec2-user -c "tail -20 /home/ec2-user/vllm.log"'
    ]}
)

time.sleep(5)

check_result = ssm.get_command_invocation(
    CommandId=check_response['Command']['CommandId'],
    InstanceId=INSTANCE_ID
)

print(f"\n{check_result['StandardOutputContent']}")

print(f"\n✅ vLLM GPU Instance Ready!")
print(f"   IP: {GPU_IP}")
print(f"   Port: 8000")
print(f"   Model: Llama 3.1 8B")

