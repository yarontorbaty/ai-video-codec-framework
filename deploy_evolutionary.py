#!/usr/bin/env python3
"""Deploy evolutionary orchestrator to EC2"""

import boto3
import time

# Configuration
INSTANCE_ID = 'i-0ee283400d2e131a4'
REGION = 'us-east-1'
WORKER_IP = '172.31.65.58'

# Initialize SSM client
ssm = boto3.client('ssm', region_name=REGION)

# Read the evolutionary orchestrator code
with open('v3/orchestrator/evolutionary_orchestrator.py', 'r') as f:
    code = f.read()

print("🚀 Deploying evolutionary orchestrator...")

# Upload code to /tmp first
commands = f"""
pkill -f orchestrator.py || true
mkdir -p /home/ec2-user/evolutionary-orchestrator
cat > /home/ec2-user/evolutionary-orchestrator/evolutionary_orchestrator.py << 'EOFPYTHON'
{code}
EOFPYTHON
cd /home/ec2-user/evolutionary-orchestrator
export WORKER_URL=http://{WORKER_IP}:8080
nohup python3 evolutionary_orchestrator.py > evolutionary.log 2>&1 &
sleep 3
ps aux | grep evolutionary_orchestrator.py | grep -v grep || echo "Not running"
tail -20 evolutionary.log || echo "No log yet"
"""

# Send command
response = ssm.send_command(
    InstanceIds=[INSTANCE_ID],
    DocumentName='AWS-RunShellScript',
    Parameters={'commands': [commands]}
)

command_id = response['Command']['CommandId']
print(f"📤 Command ID: {command_id}")
print("⏳ Waiting for completion...")

# Wait for command to complete
time.sleep(10)

# Get output
result = ssm.get_command_invocation(
    CommandId=command_id,
    InstanceId=INSTANCE_ID
)

print(f"\n✅ Status: {result['Status']}")
print(f"\n📊 Output:\n{result['StandardOutputContent']}")

if result['StandardErrorContent']:
    print(f"\n❌ Errors:\n{result['StandardErrorContent']}")

