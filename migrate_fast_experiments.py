#!/usr/bin/env python3
"""
Migrate fast experiments to match original schema for dashboard compatibility
"""

import boto3
import math
from decimal import Decimal
from typing import Dict, Any

# Configuration
TABLE_NAME = 'ai-codec-v3-fast-experiments'
REGION = 'us-east-1'

# Video specs for fast experiments
VIDEO_WIDTH = 64
VIDEO_HEIGHT = 64
VIDEO_FRAMES = 10
VIDEO_FPS = 30
ORIGINAL_SIZE = VIDEO_WIDTH * VIDEO_HEIGHT * 3 * VIDEO_FRAMES  # 122,880 bytes

# Initialize DynamoDB
dynamodb = boto3.resource('dynamodb', region_name=REGION)
table = dynamodb.Table(TABLE_NAME)


def mse_to_psnr(mse: float) -> float:
    """Convert MSE to PSNR in dB"""
    if mse == 0:
        return 100.0  # Perfect quality
    max_pixel_value = 255.0
    psnr = 10 * math.log10((max_pixel_value ** 2) / mse)
    return psnr


def estimate_ssim_from_psnr(psnr: float) -> float:
    """Estimate SSIM from PSNR (rough approximation)"""
    # PSNR to SSIM correlation (approximate)
    # PSNR 20-30 dB => SSIM 0.6-0.8
    # PSNR 30-40 dB => SSIM 0.8-0.95
    # PSNR 40+ dB => SSIM 0.95-1.0
    if psnr >= 40:
        return min(0.95 + (psnr - 40) * 0.01, 1.0)
    elif psnr >= 30:
        return 0.8 + (psnr - 30) * 0.015
    elif psnr >= 20:
        return 0.6 + (psnr - 20) * 0.02
    else:
        return max(0.3 + (psnr - 10) * 0.03, 0.0)


def calculate_bitrate(compressed_size: int) -> float:
    """Calculate bitrate in Mbps"""
    duration_seconds = VIDEO_FRAMES / VIDEO_FPS  # ~0.333 seconds
    bitrate_bps = (compressed_size * 8) / duration_seconds
    bitrate_mbps = bitrate_bps / 1_000_000
    return bitrate_mbps


def migrate_experiment(item: Dict[str, Any]) -> Dict[str, Any]:
    """Transform fast experiment to original schema"""
    
    # Extract fast experiment fields
    mse = float(item.get('mse', 0))
    compression_ratio = float(item.get('compression_ratio', 1.0))
    compressed_size = int(item.get('compressed_size', ORIGINAL_SIZE))
    time_ms = int(item.get('time_ms', 0))
    status = item.get('status', 'unknown')
    experiment_id = item.get('experiment_id', '')
    timestamp = int(item.get('timestamp', 0))
    generation = int(item.get('generation', 0))
    
    # Calculate metrics
    psnr = mse_to_psnr(mse) if mse > 0 else 0.0
    ssim = estimate_ssim_from_psnr(psnr) if psnr > 0 else 0.0
    bitrate = calculate_bitrate(compressed_size)
    
    # Create updated item with original schema
    updated_item = {
        'experiment_id': experiment_id,
        'timestamp': timestamp,
        'status': status,
        'metrics': {
            'psnr_db': Decimal(str(round(psnr, 2))),
            'ssim': Decimal(str(round(ssim, 4))),
            'bitrate_mbps': Decimal(str(round(bitrate, 3))),
            'mse': Decimal(str(round(mse, 2))),
            'compression_ratio': Decimal(str(round(compression_ratio, 2))),
            'encoding_time_ms': time_ms,
            'compressed_size_bytes': compressed_size,
        }
    }
    
    # Add generation if present
    if generation > 0:
        updated_item['generation'] = generation
    
    # Add error field if failed
    if status == 'failed' and 'error' in item:
        updated_item['error'] = item['error']
    
    return updated_item


def migrate_all_experiments():
    """Migrate all experiments in the table"""
    
    print("🔄 Starting migration of fast experiments...")
    print(f"📊 Table: {TABLE_NAME}\n")
    
    # Scan all experiments
    response = table.scan()
    experiments = response.get('Items', [])
    
    # Handle pagination
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        experiments.extend(response.get('Items', []))
    
    total = len(experiments)
    print(f"📦 Found {total} experiments to migrate\n")
    
    # Migrate in batches
    batch_size = 25  # DynamoDB batch write limit
    migrated = 0
    errors = 0
    
    for i in range(0, total, batch_size):
        batch = experiments[i:i+batch_size]
        
        with table.batch_writer() as writer:
            for item in batch:
                try:
                    updated_item = migrate_experiment(item)
                    writer.put_item(Item=updated_item)
                    migrated += 1
                    
                    if migrated % 100 == 0:
                        print(f"✅ Migrated {migrated}/{total} experiments...")
                        
                except Exception as e:
                    errors += 1
                    print(f"❌ Error migrating {item.get('experiment_id', 'unknown')}: {e}")
    
    print(f"\n🎉 Migration complete!")
    print(f"   ✅ Migrated: {migrated}")
    print(f"   ❌ Errors: {errors}")
    print(f"\n📊 Sample migrated experiment:")
    
    # Show a sample
    if experiments:
        sample = migrate_experiment(experiments[0])
        print(f"   Experiment ID: {sample['experiment_id']}")
        print(f"   Status: {sample['status']}")
        print(f"   Metrics:")
        for key, value in sample['metrics'].items():
            print(f"      {key}: {value}")


if __name__ == '__main__':
    migrate_all_experiments()

