#!/usr/bin/env python3
"""
Purge all experiments from DynamoDB tables.
This clears all historical data to start fresh.
"""

import boto3
from decimal import Decimal

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

# Tables to purge
TABLES = [
    'ai-video-codec-experiments',
    'ai-video-codec-metrics', 
    'ai-video-codec-reasoning'
]

def purge_table(table_name):
    """Delete all items from a DynamoDB table."""
    table = dynamodb.Table(table_name)
    
    print(f"\nPurging table: {table_name}")
    
    # Get table key schema
    key_schema = table.key_schema
    key_names = [key['AttributeName'] for key in key_schema]
    
    print(f"  Key attributes: {key_names}")
    
    # Scan and delete all items
    response = table.scan()
    items = response.get('Items', [])
    
    if not items:
        print(f"  ✅ Table is already empty")
        return
    
    print(f"  Found {len(items)} items to delete...")
    
    deleted = 0
    with table.batch_writer() as batch:
        for item in items:
            # Extract only the key attributes
            key = {k: item[k] for k in key_names if k in item}
            batch.delete_item(Key=key)
            deleted += 1
            if deleted % 10 == 0:
                print(f"  Deleted {deleted}/{len(items)}...")
    
    print(f"  ✅ Deleted {deleted} items from {table_name}")

def main():
    print("=" * 60)
    print("PURGING ALL EXPERIMENT DATA")
    print("=" * 60)
    print("\nThis will delete all data from:")
    for table in TABLES:
        print(f"  - {table}")
    print()
    
    for table_name in TABLES:
        try:
            purge_table(table_name)
        except Exception as e:
            print(f"  ❌ Error purging {table_name}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ PURGE COMPLETE - All experiments cleared")
    print("=" * 60)
    print("\nThe orchestrator will start generating new experiments")
    print("with real LLM analysis from scratch.\n")

if __name__ == '__main__':
    main()

