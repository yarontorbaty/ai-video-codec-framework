#!/usr/bin/env python3
"""
Real-time monitoring dashboard for fast experiments
Shows improvements in compression and quality
"""

import boto3
import time
import os
from decimal import Decimal
from collections import defaultdict

# Initialize DynamoDB
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('ai-codec-v3-fast-experiments')

def get_all_experiments():
    """Fetch all experiments from DynamoDB"""
    response = table.scan()
    items = response['Items']
    
    # Handle pagination
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response['Items'])
    
    return items

def analyze_experiments(experiments):
    """Analyze experiments and find improvements"""
    
    # Separate by status
    successful = [e for e in experiments if e.get('status') == 'success']
    failed = [e for e in experiments if e.get('status') == 'failed']
    
    if not successful:
        return None
    
    # Convert Decimal to float for calculations
    for exp in successful:
        if 'mse' in exp:
            exp['mse'] = float(exp['mse'])
        if 'compression_ratio' in exp:
            exp['compression_ratio'] = float(exp['compression_ratio'])
        if 'time_ms' in exp:
            exp['time_ms'] = int(exp['time_ms'])
        if 'compressed_size' in exp:
            exp['compressed_size'] = int(exp['compressed_size'])
    
    # Sort by compression ratio (higher is better)
    by_compression = sorted(successful, key=lambda x: x.get('compression_ratio', 0), reverse=True)
    
    # Sort by MSE (lower is better - better quality)
    by_quality = sorted(successful, key=lambda x: x.get('mse', float('inf')))
    
    # Calculate statistics
    compression_ratios = [e['compression_ratio'] for e in successful if 'compression_ratio' in e]
    mse_values = [e['mse'] for e in successful if 'mse' in e]
    time_values = [e['time_ms'] for e in successful if 'time_ms' in e]
    
    stats = {
        'total': len(experiments),
        'successful': len(successful),
        'failed': len(failed),
        'success_rate': len(successful) / len(experiments) * 100 if experiments else 0,
        'avg_compression_ratio': sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0,
        'max_compression_ratio': max(compression_ratios) if compression_ratios else 0,
        'min_compression_ratio': min(compression_ratios) if compression_ratios else 0,
        'avg_mse': sum(mse_values) / len(mse_values) if mse_values else 0,
        'min_mse': min(mse_values) if mse_values else 0,
        'max_mse': max(mse_values) if mse_values else 0,
        'avg_time_ms': sum(time_values) / len(time_values) if time_values else 0,
        'top_compression': by_compression[:5] if len(by_compression) >= 5 else by_compression,
        'best_quality': by_quality[:5] if len(by_quality) >= 5 else by_quality,
    }
    
    # Analyze failure reasons
    failure_reasons = defaultdict(int)
    for exp in failed:
        error = exp.get('error', 'Unknown error')
        # Categorize errors
        if 'Encoding failed' in error:
            failure_reasons['Encoding errors'] += 1
        elif 'Decoding failed' in error:
            failure_reasons['Decoding errors'] += 1
        elif 'timeout' in error.lower():
            failure_reasons['Timeouts'] += 1
        elif 'import' in error.lower():
            failure_reasons['Import errors'] += 1
        else:
            failure_reasons['Other errors'] += 1
    
    stats['failure_reasons'] = dict(failure_reasons)
    
    return stats

def print_dashboard(stats, iteration=0):
    """Print a beautiful dashboard"""
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print("=" * 80)
    print("🚀 FAST EXPERIMENT SYSTEM - REAL-TIME MONITORING")
    print("=" * 80)
    print(f"Refresh #{iteration} - {time.strftime('%H:%M:%S')}")
    print()
    
    if not stats:
        print("⏳ No experiments yet...")
        return
    
    # Overall stats
    print(f"📊 OVERALL STATISTICS")
    print(f"   Total Experiments: {stats['total']}")
    print(f"   ✅ Successful: {stats['successful']} ({stats['success_rate']:.1f}%)")
    print(f"   ❌ Failed: {stats['failed']} ({100-stats['success_rate']:.1f}%)")
    print()
    
    # Compression stats
    print(f"🗜️  COMPRESSION PERFORMANCE")
    print(f"   Average Ratio: {stats['avg_compression_ratio']:.2f}x")
    print(f"   Best Ratio: {stats['max_compression_ratio']:.2f}x")
    print(f"   Worst Ratio: {stats['min_compression_ratio']:.2f}x")
    print()
    
    # Quality stats
    print(f"🎨 QUALITY METRICS (MSE - lower is better)")
    print(f"   Average MSE: {stats['avg_mse']:.2f}")
    print(f"   Best MSE: {stats['min_mse']:.2f}")
    print(f"   Worst MSE: {stats['max_mse']:.2f}")
    print()
    
    # Speed stats
    print(f"⚡ PROCESSING SPEED")
    print(f"   Average Time: {stats['avg_time_ms']:.1f}ms per experiment")
    print()
    
    # Top performers
    print(f"🏆 TOP 5 COMPRESSION LEADERS")
    for i, exp in enumerate(stats['top_compression'], 1):
        exp_id = exp['experiment_id'][-30:]  # Last 30 chars
        print(f"   {i}. {exp_id}")
        print(f"      Compression: {exp['compression_ratio']:.2f}x | MSE: {exp['mse']:.2f} | Time: {exp.get('time_ms', 0)}ms")
    print()
    
    print(f"✨ TOP 5 QUALITY LEADERS (Lowest MSE)")
    for i, exp in enumerate(stats['best_quality'], 1):
        exp_id = exp['experiment_id'][-30:]
        print(f"   {i}. {exp_id}")
        print(f"      MSE: {exp['mse']:.2f} | Compression: {exp['compression_ratio']:.2f}x | Time: {exp.get('time_ms', 0)}ms")
    print()
    
    # Failure analysis
    if stats['failure_reasons']:
        print(f"🔍 FAILURE BREAKDOWN")
        for reason, count in sorted(stats['failure_reasons'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {reason}: {count}")
        print()
    
    print("=" * 80)
    print("Press Ctrl+C to stop monitoring...")
    print("=" * 80)

def main():
    """Main monitoring loop"""
    iteration = 0
    
    try:
        while True:
            iteration += 1
            
            # Fetch and analyze experiments
            experiments = get_all_experiments()
            stats = analyze_experiments(experiments)
            
            # Display dashboard
            print_dashboard(stats, iteration)
            
            # Wait before refresh
            time.sleep(5)  # Refresh every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped.")
        if stats:
            print(f"\n📊 Final Stats: {stats['successful']} successful, {stats['failed']} failed")
            print(f"   Best compression: {stats['max_compression_ratio']:.2f}x")
            print(f"   Best quality: {stats['min_mse']:.2f} MSE")

if __name__ == '__main__':
    main()

