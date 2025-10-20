#!/usr/bin/env python3
"""
Monitor evolutionary orchestrator progress and notify when complete
"""

import boto3
import time
from datetime import datetime

# Configuration
ORCHESTRATOR_INSTANCE = 'i-0ee283400d2e131a4'
DYNAMODB_TABLE = 'ai-codec-v3-fast-experiments'
REGION = 'us-east-1'
TARGET_GENERATIONS = 50
CHECK_INTERVAL = 300  # 5 minutes

# Initialize clients
ssm = boto3.client('ssm', region_name=REGION)
dynamodb = boto3.resource('dynamodb', region_name=REGION)
table = dynamodb.Table(DYNAMODB_TABLE)

def get_experiment_stats():
    """Get current experiment statistics"""
    response = table.scan(Select='COUNT')
    total_count = response['Count']
    
    # Get count by generation (sample to avoid full scan)
    response = table.scan(
        ProjectionExpression='generation',
        Limit=1000
    )
    
    generations = {}
    for item in response.get('Items', []):
        gen = int(item.get('generation', 0))
        generations[gen] = generations.get(gen, 0) + 1
    
    max_gen = max(generations.keys()) if generations else 0
    
    return {
        'total': total_count,
        'max_generation': max_gen,
        'generations': generations
    }

def get_orchestrator_logs(lines=30):
    """Get latest orchestrator logs"""
    try:
        response = ssm.send_command(
            InstanceIds=[ORCHESTRATOR_INSTANCE],
            DocumentName='AWS-RunShellScript',
            Parameters={'commands': [
                f'tail -{lines} /home/ec2-user/evolutionary-orchestrator/evolutionary.log 2>/dev/null || echo "No logs yet"'
            ]}
        )
        
        command_id = response['Command']['CommandId']
        time.sleep(3)
        
        result = ssm.get_command_invocation(
            CommandId=command_id,
            InstanceId=ORCHESTRATOR_INSTANCE
        )
        
        return result['StandardOutputContent']
    except Exception as e:
        return f"Error getting logs: {e}"

def print_status(stats, iteration):
    """Print current status"""
    print(f"\n{'='*80}")
    print(f"🧬 EVOLUTIONARY SYSTEM MONITOR - Check #{iteration}")
    print(f"{'='*80}")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📊 EXPERIMENT STATISTICS:")
    print(f"   Total Experiments: {stats['total']:,}")
    print(f"   Current Generation: {stats['max_generation']}/{TARGET_GENERATIONS}")
    print(f"   Progress: {(stats['max_generation']/TARGET_GENERATIONS*100):.1f}%")
    
    if stats['generations']:
        print(f"\n🧬 GENERATION BREAKDOWN:")
        for gen in sorted(stats['generations'].keys())[-5:]:
            count = stats['generations'][gen]
            print(f"   Gen {gen}: {count} experiments")
    
    # Estimate completion
    if stats['max_generation'] > 0:
        gens_remaining = TARGET_GENERATIONS - stats['max_generation']
        minutes_remaining = gens_remaining * 5  # ~5 min per generation
        hours = minutes_remaining // 60
        mins = minutes_remaining % 60
        print(f"\n⏱️  ESTIMATED TIME REMAINING: {hours}h {mins}m")
    
    print(f"\n{'='*80}")

def monitor():
    """Main monitoring loop"""
    print("🚀 Starting evolutionary system monitor...")
    print(f"Target: {TARGET_GENERATIONS} generations")
    print(f"Checking every {CHECK_INTERVAL//60} minutes\n")
    
    iteration = 1
    start_gen = None
    
    while True:
        stats = get_experiment_stats()
        
        if start_gen is None:
            start_gen = stats['max_generation']
        
        print_status(stats, iteration)
        
        # Check if complete
        if stats['max_generation'] >= TARGET_GENERATIONS:
            print("\n" + "🎉" * 40)
            print("✅ EVOLUTIONARY RUN COMPLETE!")
            print("🎉" * 40)
            print(f"\n📈 FINAL STATISTICS:")
            print(f"   Total Experiments: {stats['total']:,}")
            print(f"   Generations Completed: {stats['max_generation']}")
            print(f"   Generations This Run: {stats['max_generation'] - start_gen}")
            print(f"\n🌐 View results at: https://aiv1codec.com")
            print(f"\n📋 Get orchestrator logs:")
            print(f"   aws ssm send-command --instance-ids {ORCHESTRATOR_INSTANCE} \\")
            print(f"     --document-name AWS-RunShellScript \\")
            print(f"     --parameters 'commands=[\"tail -100 /home/ec2-user/evolutionary-orchestrator/evolutionary.log\"]' \\")
            print(f"     --region {REGION}")
            break
        
        # Show recent logs
        if iteration % 2 == 0:  # Every other check
            print("\n📋 RECENT ORCHESTRATOR LOGS:")
            logs = get_orchestrator_logs(20)
            print(logs[-1000:])  # Last 1000 chars
        
        print(f"\n💤 Sleeping {CHECK_INTERVAL//60} minutes until next check...")
        print("   (Press Ctrl+C to stop monitoring)")
        
        time.sleep(CHECK_INTERVAL)
        iteration += 1

if __name__ == '__main__':
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoring stopped by user")
        print("   Claude orchestrator is still running!")
        print(f"   Resume monitoring: python3 {__file__}")

