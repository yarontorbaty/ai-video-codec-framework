#!/usr/bin/env python3
"""
Purge all experiments that used the fallback baseline approach (15.04 Mbps bitrate).
"""

import boto3
import json
from decimal import Decimal

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
experiments_table = dynamodb.Table('ai-video-codec-experiments')

def scan_fallback_experiments():
    """Find all experiments with 15.04 Mbps bitrate."""
    response = experiments_table.scan()
    items = response.get('Items', [])
    
    # Handle pagination
    while 'LastEvaluatedKey' in response:
        response = experiments_table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response.get('Items', []))
    
    fallback_experiments = []
    
    for item in items:
        try:
            experiments = json.loads(item.get('experiments', '[]'))
            
            # Check if any experiment has bitrate between 15.0 and 15.1 Mbps
            for exp in experiments:
                bitrate = exp.get('real_metrics', {}).get('bitrate_mbps', 0)
                if 15.0 < bitrate < 15.1:
                    fallback_experiments.append(item)
                    break
        except:
            pass
    
    return fallback_experiments

def delete_experiment(experiment_id, timestamp):
    """Delete an experiment from DynamoDB."""
    try:
        experiments_table.delete_item(
            Key={
                'experiment_id': experiment_id,
                'timestamp': int(timestamp)
            }
        )
        print(f"  ✅ Deleted {experiment_id}")
        return True
    except Exception as e:
        print(f"  ❌ Failed to delete {experiment_id}: {e}")
        return False

def main():
    print("🔍 Scanning for fallback experiments (15.04 Mbps)...")
    
    fallback_experiments = scan_fallback_experiments()
    
    print(f"\n📊 Found {len(fallback_experiments)} fallback experiments")
    
    if not fallback_experiments:
        print("✅ No fallback experiments to purge")
        return
    
    print("\nExperiments to delete:")
    for exp in fallback_experiments:
        exp_id = exp['experiment_id']
        timestamp = exp['timestamp']
        print(f"  - {exp_id} (timestamp: {timestamp})")
    
    print(f"\n🗑️  Deleting {len(fallback_experiments)} experiments...")
    
    deleted_count = 0
    for exp in fallback_experiments:
        exp_id = exp['experiment_id']
        timestamp = exp['timestamp']
        
        if delete_experiment(exp_id, timestamp):
            deleted_count += 1
    
    print(f"\n✅ Purge complete!")
    print(f"   Deleted: {deleted_count}/{len(fallback_experiments)} experiments")
    
    # Also try to delete from reasoning table
    try:
        reasoning_table = dynamodb.Table('ai-video-codec-reasoning')
        print("\n🔍 Cleaning up reasoning table...")
        
        for exp in fallback_experiments:
            exp_id = exp['experiment_id']
            try:
                reasoning_table.delete_item(Key={'reasoning_id': exp_id})
                print(f"  ✅ Deleted reasoning for {exp_id}")
            except:
                pass
    except Exception as e:
        print(f"  Note: Could not clean reasoning table: {e}")

if __name__ == '__main__':
    main()

