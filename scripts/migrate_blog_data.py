#!/usr/bin/env python3
"""
Migrate existing experiments to blog-compatible format.
Fixes experiments that were stored in old format without proper blog structure.
"""

import boto3
import json
from datetime import datetime
from decimal import Decimal

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
experiments_table = dynamodb.Table('ai-video-codec-experiments')

def migrate_experiment(exp):
    """
    Transform experiment to blog-compatible format.
    
    Args:
        exp: Experiment item from DynamoDB
        
    Returns:
        Updated experiment item, or None if no changes needed
    """
    exp_id = exp.get('experiment_id', '')
    
    # Check if already in new format
    experiments_field = exp.get('experiments')
    if experiments_field:
        # Already has experiments field - check if it's valid
        try:
            if isinstance(experiments_field, str):
                experiments_array = json.loads(experiments_field)
                if experiments_array and isinstance(experiments_array, list):
                    print(f"  ✓ {exp_id}: Already in new format")
                    return None
        except:
            pass
    
    # Extract metrics from old format
    real_metrics = exp.get('real_metrics', {})
    
    # If no metrics, check if stored differently
    if not real_metrics:
        # Try to extract from experiment_data if it exists
        print(f"  ⚠️  {exp_id}: No metrics found, skipping")
        return None
    
    # Convert Decimal to float for JSON serialization
    def decimal_to_float(obj):
        if isinstance(obj, dict):
            return {k: decimal_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [decimal_to_float(item) for item in obj]
        elif isinstance(obj, Decimal):
            return float(obj)
        return obj
    
    real_metrics = decimal_to_float(real_metrics)
    
    # Calculate HEVC comparison
    hevc_baseline_mbps = 10.0
    bitrate_mbps = real_metrics.get('bitrate_mbps', 0)
    reduction_percent = ((hevc_baseline_mbps - bitrate_mbps) / hevc_baseline_mbps) * 100 if bitrate_mbps else -50.0
    target_achieved = bitrate_mbps < 1.0 if bitrate_mbps else False
    
    # Create experiments array
    experiments_array = [{
        'experiment_type': 'real_procedural_generation',
        'status': 'completed',
        'real_metrics': real_metrics,
        'comparison': {
            'hevc_baseline_mbps': hevc_baseline_mbps,
            'reduction_percent': reduction_percent,
            'target_achieved': target_achieved
        }
    }]
    
    # Create timestamp_iso if missing
    timestamp = exp.get('timestamp', 0)
    timestamp_iso = exp.get('timestamp_iso')
    if not timestamp_iso and timestamp:
        try:
            timestamp_iso = datetime.utcfromtimestamp(int(timestamp)).isoformat() + 'Z'
        except:
            timestamp_iso = datetime.utcnow().isoformat() + 'Z'
    
    # Build update
    updates = {
        'experiments': json.dumps(experiments_array)
    }
    
    if timestamp_iso:
        updates['timestamp_iso'] = timestamp_iso
    
    print(f"  ✅ {exp_id}: Migrating (bitrate: {bitrate_mbps:.2f} Mbps, reduction: {reduction_percent:.1f}%)")
    return updates


def main():
    """Migrate all experiments to blog-compatible format."""
    print("=" * 60)
    print("MIGRATING EXPERIMENTS TO BLOG-COMPATIBLE FORMAT")
    print("=" * 60)
    print()
    
    # Scan all experiments
    print("📊 Scanning experiments table...")
    response = experiments_table.scan()
    items = response.get('Items', [])
    
    # Handle pagination
    while 'LastEvaluatedKey' in response:
        response = experiments_table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response.get('Items', []))
    
    print(f"Found {len(items)} total experiments")
    print()
    
    migrated = 0
    skipped = 0
    errors = 0
    
    for exp in items:
        exp_id = exp.get('experiment_id', 'unknown')
        
        try:
            updates = migrate_experiment(exp)
            
            if updates:
                # Update the item in DynamoDB
                update_expression = "SET " + ", ".join([f"#{k} = :{k}" for k in updates.keys()])
                expression_attribute_names = {f"#{k}": k for k in updates.keys()}
                expression_attribute_values = {f":{k}": v for k, v in updates.items()}
                
                experiments_table.update_item(
                    Key={'experiment_id': exp_id},
                    UpdateExpression=update_expression,
                    ExpressionAttributeNames=expression_attribute_names,
                    ExpressionAttributeValues=expression_attribute_values
                )
                
                migrated += 1
            else:
                skipped += 1
                
        except Exception as e:
            print(f"  ❌ {exp_id}: Error - {e}")
            errors += 1
    
    print()
    print("=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(f"✅ Migrated: {migrated}")
    print(f"⏭️  Skipped:  {skipped} (already in new format or no data)")
    print(f"❌ Errors:   {errors}")
    print()
    
    if migrated > 0:
        print("🎉 Blog should now show results for all experiments!")
    else:
        print("ℹ️  No experiments needed migration")


if __name__ == '__main__':
    main()

