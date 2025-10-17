#!/usr/bin/env python3
"""
Backfill blog post approaches for existing experiments.

This script fetches the reasoning data for each experiment and updates
the experiments table to include the approach/hypothesis in the experiments array.
"""

import boto3
import json
from decimal import Decimal

def decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_float(item) for item in obj]
    return obj

def convert_floats(obj):
    if isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(item) for item in obj]
    elif isinstance(obj, float):
        return Decimal(str(obj))
    return obj

def main():
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    experiments_table = dynamodb.Table('ai-video-codec-experiments')
    reasoning_table = dynamodb.Table('ai-video-codec-reasoning')
    
    print("🔧 BACKFILLING BLOG POST APPROACHES")
    print("=" * 60)
    
    # Fetch all experiments
    response = experiments_table.scan()
    experiments = response.get('Items', [])
    
    while 'LastEvaluatedKey' in response:
        response = experiments_table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        experiments.extend(response.get('Items', []))
    
    print(f"Found {len(experiments)} experiments")
    
    # Fetch all reasoning data
    response = reasoning_table.scan()
    reasoning_items = response.get('Items', [])
    
    while 'LastEvaluatedKey' in response:
        response = reasoning_table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        reasoning_items.extend(response.get('Items', []))
    
    print(f"Found {len(reasoning_items)} reasoning records")
    
    # Create reasoning lookup
    reasoning_lookup = {}
    for item in reasoning_items:
        exp_id = item.get('experiment_id')
        if exp_id:
            reasoning_lookup[exp_id] = decimal_to_float(item)
    
    print(f"Created lookup for {len(reasoning_lookup)} experiments")
    print()
    
    updated_count = 0
    skipped_count = 0
    
    for exp in experiments:
        exp_id = exp.get('experiment_id')
        timestamp = exp.get('timestamp')
        status = exp.get('status')
        
        if status != 'completed':
            continue
        
        try:
            # Check if experiments field exists and has approach
            experiments_json = exp.get('experiments')
            if not experiments_json:
                skipped_count += 1
                continue
            
            exp_array = json.loads(experiments_json)
            if not exp_array or len(exp_array) == 0:
                skipped_count += 1
                continue
            
            # Check if already has approach
            if exp_array[0].get('approach') and exp_array[0]['approach'] != 'N/A':
                skipped_count += 1
                continue
            
            # Try to get approach from reasoning table
            reasoning = reasoning_lookup.get(exp_id)
            if reasoning:
                hypothesis = reasoning.get('hypothesis', '')
                if hypothesis:
                    # Update the approach field
                    exp_array[0]['approach'] = hypothesis
                    
                    # Update in DynamoDB
                    experiments_table.update_item(
                        Key={
                            'experiment_id': exp_id,
                            'timestamp': timestamp
                        },
                        UpdateExpression='SET experiments = :exp',
                        ExpressionAttributeValues={
                            ':exp': json.dumps(exp_array)
                        }
                    )
                    
                    print(f"✅ {exp_id}: Updated with approach")
                    updated_count += 1
                else:
                    print(f"⚠️  {exp_id}: No hypothesis in reasoning")
                    skipped_count += 1
            else:
                print(f"⚠️  {exp_id}: No reasoning data found")
                skipped_count += 1
                
        except Exception as e:
            print(f"❌ {exp_id}: Error - {e}")
            skipped_count += 1
    
    print()
    print("=" * 60)
    print(f"✅ Updated: {updated_count}")
    print(f"⚠️  Skipped: {skipped_count}")
    print(f"📊 Total: {len(experiments)}")

if __name__ == '__main__':
    main()

