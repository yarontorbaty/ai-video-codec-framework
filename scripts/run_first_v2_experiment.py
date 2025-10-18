#!/usr/bin/env python3
"""
Quick start script to run your first v2.0 neural codec experiment
"""

import os
import sys
import boto3
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.gpu_first_orchestrator import GPUFirstOrchestrator


def main():
    print("=" * 80)
    print("🚀 v2.0 NEURAL CODEC - FIRST EXPERIMENT")
    print("=" * 80)
    print()
    
    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ERROR: ANTHROPIC_API_KEY environment variable not set")
        print()
        print("Please set your Anthropic API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key-here'")
        print()
        sys.exit(1)
    
    # Configuration
    config = {
        "anthropic_api_key": api_key,
        "experiment_table": "ai-video-codec-experiments",
        "training_queue_url": "https://sqs.us-east-1.amazonaws.com/580473065386/ai-video-codec-training-queue",
        "s3_bucket": "ai-video-codec-videos-580473065386",
        "region": "us-east-1"
    }
    
    print("📋 Configuration:")
    print(f"   DynamoDB Table: {config['experiment_table']}")
    print(f"   SQS Queue: {config['training_queue_url'].split('/')[-1]}")
    print(f"   S3 Bucket: {config['s3_bucket']}")
    print(f"   Region: {config['region']}")
    print()
    
    # Check GPU worker status
    print("🔍 Checking GPU worker status...")
    ssm = boto3.client('ssm', region_name=config['region'])
    try:
        response = ssm.send_command(
            InstanceIds=['i-0b614aa221757060e'],
            DocumentName="AWS-RunShellScript",
            Parameters={
                'commands': [
                    'ps aux | grep neural_codec_gpu_worker | grep -v grep || echo "No worker running"',
                    'tail -5 /tmp/gpu_worker.log 2>/dev/null || echo "No logs yet"'
                ]
            }
        )
        print("   ✅ Worker check initiated (command ID: {})".format(response['Command']['CommandId']))
    except Exception as e:
        print(f"   ⚠️  Could not check worker status: {e}")
    print()
    
    # Initialize orchestrator
    print("🎯 Initializing GPU-First Orchestrator...")
    try:
        orchestrator = GPUFirstOrchestrator(
            anthropic_api_key=config['anthropic_api_key'],
            experiment_table=config['experiment_table'],
            training_queue_url=config['training_queue_url'],
            s3_bucket=config['s3_bucket'],
            region=config['region']
        )
        print("   ✅ Orchestrator initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize orchestrator: {e}")
        sys.exit(1)
    print()
    
    # Start experiment
    experiment_goal = (
        "Compress test video achieving 90% bitrate reduction while maintaining "
        ">95% quality preservation. Use the two-agent neural codec approach with "
        "scene-adaptive compression strategy selection."
    )
    
    video_key = "test_data/HEVC_HD_10Mbps.mp4"
    
    print("🎬 Starting Experiment:")
    print(f"   Goal: {experiment_goal}")
    print(f"   Video: {video_key}")
    print()
    
    try:
        experiment_id = orchestrator.start_experiment(
            goal=experiment_goal,
            video_key=video_key
        )
        
        print("=" * 80)
        print("✅ EXPERIMENT STARTED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print(f"📊 Experiment ID: {experiment_id}")
        print()
        print("Monitor progress:")
        print(f"  - DynamoDB: {config['experiment_table']}")
        print(f"  - SQS Queue: {config['training_queue_url'].split('/')[-1]}")
        print()
        print("Check GPU worker logs:")
        print("  aws ssm send-command \\")
        print("      --instance-ids i-0b614aa221757060e \\")
        print("      --document-name 'AWS-RunShellScript' \\")
        print("      --parameters 'commands=[\"tail -50 /tmp/gpu_worker.log\"]'")
        print()
        print("View experiment status:")
        print(f"  aws dynamodb get-item \\")
        print(f"      --table-name {config['experiment_table']} \\")
        print(f"      --key '{{\"experiment_id\":{{\"S\":\"{experiment_id}\"}}}}'")
        print()
        
    except Exception as e:
        print(f"❌ Failed to start experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

