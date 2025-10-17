#!/usr/bin/env python3
"""
Consolidate Duplicate Experiment Records

Problem: Multiple records per experiment due to timestamp changes in put_item()
Solution: Keep the most complete record (preferably completed status) with earliest timestamp

This script:
1. Scans all experiments
2. Groups by experiment_id
3. For each experiment with multiple records:
   - Selects the best record (completed > running > timed_out)
   - Updates it to use the earliest timestamp
   - Deletes duplicate records
"""

import boto3
import json
from decimal import Decimal
from collections import defaultdict

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
experiments_table = dynamodb.Table('ai-video-codec-experiments')


def decimal_to_float(obj):
    """Convert Decimal to float for JSON serialization"""
    if isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_float(item) for item in obj]
    elif isinstance(obj, Decimal):
        return float(obj)
    return obj


def convert_floats(obj):
    """Convert floats to Decimal for DynamoDB"""
    if isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(item) for item in obj]
    elif isinstance(obj, float):
        return Decimal(str(obj))
    return obj


def select_best_record(records):
    """
    Select the best record from duplicates.
    
    Priority:
    1. Completed with experiments data
    2. Completed without experiments data
    3. Timed_out with experiments data
    4. Running with experiments data
    5. Latest record by phase
    """
    # Sort by preference
    def record_score(r):
        status = r.get('status', '')
        has_experiments = bool(r.get('experiments'))
        has_metrics = False
        
        if has_experiments:
            try:
                exp_data = json.loads(r['experiments'])
                if exp_data and 'real_metrics' in exp_data[0]:
                    metrics = exp_data[0]['real_metrics']
                    has_metrics = bool(metrics and 'bitrate_mbps' in metrics)
            except:
                pass
        
        # Score: higher is better
        score = 0
        
        # Status priority
        if status == 'completed':
            score += 1000
        elif status == 'timed_out':
            score += 500
        elif status == 'running':
            score += 100
        
        # Has experiments field
        if has_experiments:
            score += 50
        
        # Has actual metrics
        if has_metrics:
            score += 100
        
        # Phase completion (later phases are better)
        phase_scores = {
            'design': 1,
            'deploy': 2,
            'validation': 3,
            'execution': 4,
            'analysis': 5
        }
        phase = r.get('phase_completed', r.get('current_phase', ''))
        score += phase_scores.get(phase, 0)
        
        return score
    
    # Sort by score (highest first)
    sorted_records = sorted(records, key=record_score, reverse=True)
    return sorted_records[0]


def consolidate_experiment(exp_id, records):
    """
    Consolidate multiple records into one.
    
    Args:
        exp_id: Experiment ID
        records: List of record dicts
        
    Returns:
        (kept_timestamp, deleted_count, updated)
    """
    if len(records) <= 1:
        return None, 0, False
    
    # Find earliest timestamp
    earliest_timestamp = min(int(r.get('timestamp', 0)) for r in records)
    
    # Select best record
    best_record = select_best_record(records)
    best_record = decimal_to_float(best_record)
    
    print(f"\n  📦 {exp_id}:")
    print(f"     Found {len(records)} records, earliest timestamp: {earliest_timestamp}")
    print(f"     Best record: status={best_record.get('status')}, phase={best_record.get('phase_completed', 'N/A')}")
    
    # Update best record to use earliest timestamp
    best_record['timestamp'] = earliest_timestamp
    
    # Re-generate timestamp_iso if present
    if 'timestamp_iso' in best_record or best_record.get('status') == 'completed':
        from datetime import datetime
        best_record['timestamp_iso'] = datetime.utcfromtimestamp(earliest_timestamp).isoformat() + 'Z'
    
    # Write consolidated record
    try:
        experiments_table.put_item(Item=convert_floats(best_record))
        print(f"     ✅ Updated record with timestamp {earliest_timestamp}")
    except Exception as e:
        print(f"     ❌ Failed to update: {e}")
        return None, 0, False
    
    # Delete duplicate records
    deleted = 0
    for record in records:
        record_timestamp = int(record.get('timestamp', 0))
        if record_timestamp != earliest_timestamp:
            try:
                experiments_table.delete_item(
                    Key={
                        'experiment_id': exp_id,
                        'timestamp': record_timestamp
                    }
                )
                deleted += 1
            except Exception as e:
                print(f"     ⚠️  Failed to delete timestamp {record_timestamp}: {e}")
    
    print(f"     🗑️  Deleted {deleted} duplicate record(s)")
    
    return earliest_timestamp, deleted, True


def main():
    """Consolidate all duplicate experiment records"""
    print("=" * 80)
    print("CONSOLIDATING DUPLICATE EXPERIMENT RECORDS")
    print("=" * 80)
    print()
    
    # Scan all records
    print("📊 Scanning experiments table...")
    response = experiments_table.scan()
    items = response.get('Items', [])
    
    # Handle pagination
    while 'LastEvaluatedKey' in response:
        response = experiments_table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response.get('Items', []))
    
    print(f"   Found {len(items)} total records")
    print()
    
    # Group by experiment_id
    experiments = defaultdict(list)
    for item in items:
        exp_id = item.get('experiment_id', 'unknown')
        experiments[exp_id].append(item)
    
    print(f"   Grouped into {len(experiments)} unique experiments")
    print()
    
    # Find duplicates
    duplicates = {exp_id: records for exp_id, records in experiments.items() if len(records) > 1}
    
    if not duplicates:
        print("✅ No duplicates found! All experiments have single records.")
        return
    
    print(f"🔍 Found {len(duplicates)} experiments with duplicate records")
    print()
    
    # Show summary
    total_records = sum(len(records) for records in duplicates.values())
    total_duplicates = total_records - len(duplicates)
    print(f"📋 Summary:")
    print(f"   - {len(duplicates)} experiments with duplicates")
    print(f"   - {total_records} total records")
    print(f"   - {total_duplicates} duplicate records to remove")
    print()
    
    # Consolidate each
    print("🔧 Consolidating...")
    consolidated = 0
    total_deleted = 0
    
    for exp_id, records in duplicates.items():
        _, deleted, updated = consolidate_experiment(exp_id, records)
        if updated:
            consolidated += 1
            total_deleted += deleted
    
    print()
    print("=" * 80)
    print("CONSOLIDATION COMPLETE")
    print("=" * 80)
    print(f"✅ Consolidated: {consolidated} experiments")
    print(f"🗑️  Deleted: {total_deleted} duplicate records")
    print(f"💾 Space saved: {total_deleted} records")
    print()
    print("✨ All experiments now have single, consistent records!")
    print()


if __name__ == '__main__':
    main()

