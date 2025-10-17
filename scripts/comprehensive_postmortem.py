#!/usr/bin/env python3
"""
Comprehensive post-mortem analysis combining:
1. DynamoDB experiment records (metadata, results)
2. Code attempts from orchestrator (saved LLM-generated functions)
3. Orchestrator logs (execution details)

This provides a complete work log for each experiment.
"""

import boto3
import json
import os
from decimal import Decimal
from datetime import datetime
from collections import defaultdict

def decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_float(item) for item in obj]
    return obj

def fetch_experiments_from_dynamodb():
    """Fetch all experiment metadata from DynamoDB."""
    print("📥 Fetching experiments from DynamoDB...")
    
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table('ai-video-codec-experiments')
    
    response = table.scan()
    items = response.get('Items', [])
    
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response.get('Items', []))
    
    items = [decimal_to_float(item) for item in items]
    print(f"✅ Fetched {len(items)} experiments")
    
    return items

def fetch_code_attempts_list():
    """Get list of saved code attempts from orchestrator."""
    print("\n📥 Fetching code attempts list from orchestrator...")
    
    ec2 = boto3.client('ec2', region_name='us-east-1')
    ssm = boto3.client('ssm', region_name='us-east-1')
    
    # Get orchestrator instance
    response = ec2.describe_instances(
        Filters=[
            {'Name': 'tag:Name', 'Values': ['ai-video-codec-orchestrator']},
            {'Name': 'instance-state-name', 'Values': ['running']}
        ]
    )
    
    if not response['Reservations']:
        print("⚠️  Orchestrator not running")
        return []
    
    instance_id = response['Reservations'][0]['Instances'][0]['InstanceId']
    
    # List code attempts
    cmd_response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': ['ls /tmp/code_attempts/']}
    )
    
    command_id = cmd_response['Command']['CommandId']
    
    # Wait for command
    import time
    for _ in range(10):
        time.sleep(1)
        result = ssm.get_command_invocation(
            CommandId=command_id,
            InstanceId=instance_id
        )
        if result['Status'] in ['Success', 'Failed']:
            break
    
    if result['Status'] == 'Success':
        files = result['StandardOutputContent'].strip().split('\n')
        print(f"✅ Found {len(files)} code attempts")
        return files
    else:
        print("⚠️  Could not fetch code attempts")
        return []

def build_comprehensive_report(experiments):
    """Build comprehensive report combining all data sources."""
    print("\n🔬 Building comprehensive post-mortem reports...")
    
    reports = []
    
    for exp in sorted(experiments, key=lambda x: x.get('timestamp', 0), reverse=True):
        exp_id = exp.get('experiment_id', 'unknown')
        timestamp = exp.get('timestamp', 0)
        status = exp.get('status', 'unknown')
        
        report = {
            'experiment_id': exp_id,
            'timestamp': timestamp,
            'timestamp_iso': datetime.utcfromtimestamp(timestamp).isoformat() + 'Z' if timestamp else 'N/A',
            'status': status,
            'metadata': {},
            'phases': {},
            'results': {},
            'issues': [],
            'code_generation': {}
        }
        
        # Extract metadata
        report['metadata']['phase_completed'] = exp.get('phase_completed', 'unknown')
        report['metadata']['validation_retries'] = exp.get('validation_retries', 0)
        report['metadata']['execution_retries'] = exp.get('execution_retries', 0)
        report['metadata']['needs_human'] = exp.get('needs_human', False)
        
        # Extract results
        try:
            experiments_data = json.loads(exp.get('experiments', '[]'))
            if experiments_data and len(experiments_data) > 0:
                exp_data = experiments_data[0]
                
                report['results']['bitrate_mbps'] = exp_data.get('real_metrics', {}).get('bitrate_mbps', 0)
                report['results']['file_size_mb'] = exp_data.get('real_metrics', {}).get('file_size_mb', 0)
                report['results']['approach'] = exp_data.get('approach', 'N/A')
                report['results']['expected_bitrate'] = exp_data.get('expected_bitrate', 0)
                
                comparison = exp_data.get('comparison', {})
                report['results']['reduction_percent'] = comparison.get('reduction_percent', 0)
                report['results']['target_achieved'] = comparison.get('target_achieved', False)
        except Exception as e:
            report['results']['parse_error'] = str(e)
        
        # Extract issues
        if exp.get('human_intervention_reasons'):
            report['issues'] = exp['human_intervention_reasons']
        
        reports.append(report)
    
    return reports

def print_summary_report(reports):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE POST-MORTEM SUMMARY")
    print("=" * 80)
    
    total = len(reports)
    completed = sum(1 for r in reports if r['status'] == 'completed')
    running = sum(1 for r in reports if r['status'] == 'running')
    needs_human = sum(1 for r in reports if r['metadata']['needs_human'])
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"  Total experiments: {total}")
    print(f"  Completed: {completed} ({completed/total*100:.1f}%)")
    print(f"  Running: {running}")
    print(f"  Needs human intervention: {needs_human} ({needs_human/total*100:.1f}%)")
    
    # Phase completion
    print(f"\n📊 FINAL PHASES REACHED:")
    phases = defaultdict(int)
    for r in reports:
        phase = r['metadata']['phase_completed']
        phases[phase] += 1
    
    for phase, count in sorted(phases.items()):
        print(f"  {phase}: {count} ({count/total*100:.1f}%)")
    
    # Retry statistics
    print(f"\n🔄 RETRY STATISTICS:")
    val_retries = [r['metadata']['validation_retries'] for r in reports if r['metadata']['validation_retries'] > 0]
    exec_retries = [r['metadata']['execution_retries'] for r in reports if r['metadata']['execution_retries'] > 0]
    
    print(f"  Experiments with validation retries: {len(val_retries)}")
    if val_retries:
        print(f"    Average: {sum(val_retries)/len(val_retries):.1f}")
        print(f"    Max: {max(val_retries)}")
    
    print(f"  Experiments with execution retries: {len(exec_retries)}")
    if exec_retries:
        print(f"    Average: {sum(exec_retries)/len(exec_retries):.1f}")
        print(f"    Max: {max(exec_retries)}")
    
    # Results analysis
    print(f"\n📈 RESULTS ANALYSIS:")
    bitrates = [r['results'].get('bitrate_mbps', 0) for r in reports 
                if r['results'].get('bitrate_mbps', 0) > 0]
    
    if bitrates:
        bitrates_sorted = sorted(bitrates)
        print(f"  Experiments with results: {len(bitrates)}")
        print(f"  Best (lowest): {min(bitrates):.2f} Mbps")
        print(f"  Median: {bitrates_sorted[len(bitrates_sorted)//2]:.2f} Mbps")
        print(f"  Average: {sum(bitrates)/len(bitrates):.2f} Mbps")
        print(f"  Worst (highest): {max(bitrates):.2f} Mbps")
        
        # Target achievement
        target_achieved = sum(1 for b in bitrates if b < 1.0)
        baseline_beat = sum(1 for b in bitrates if b < 10.0)
        
        print(f"\n  🎯 Target (< 1 Mbps): {target_achieved}/{len(bitrates)} ({target_achieved/len(bitrates)*100:.1f}%)")
        print(f"  ✅ Beat baseline (< 10 Mbps): {baseline_beat}/{len(bitrates)} ({baseline_beat/len(bitrates)*100:.1f}%)")
        
        # Distribution
        print(f"\n  📊 BITRATE DISTRIBUTION:")
        ranges = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 50), (50, 200)]
        for low, high in ranges:
            count = sum(1 for b in bitrates if low <= b < high)
            if count > 0:
                bar = '█' * int(count * 40 / len(bitrates))
                pct = count / len(bitrates) * 100
                print(f"    {low:3d}-{high:3d} Mbps: {bar} {count:3d} ({pct:5.1f}%)")
    
    # Common issues
    print(f"\n❌ COMMON ISSUES:")
    issue_types = defaultdict(int)
    for r in reports:
        for issue in r['issues']:
            if isinstance(issue, dict):
                reason = issue.get('reason', 'unknown')
                phase = issue.get('phase', 'unknown')
                issue_types[f"{phase}: {reason}"] += 1
            elif isinstance(issue, str):
                issue_types[issue] += 1
    
    if issue_types:
        for issue, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {count}x: {issue}")
    else:
        print(f"  No issues recorded!")
    
    # Top performers
    print(f"\n🏆 TOP 10 EXPERIMENTS (Lowest Bitrate):")
    performers = [(r['experiment_id'], r['results'].get('bitrate_mbps', 999), 
                   r['results'].get('reduction_percent', 0), r['results'].get('approach', 'N/A')[:50])
                  for r in reports if r['results'].get('bitrate_mbps', 0) > 0]
    performers.sort(key=lambda x: x[1])
    
    for i, (exp_id, bitrate, reduction, approach) in enumerate(performers[:10], 1):
        icon = "🏆" if i == 1 else "✅"
        print(f"  {icon} {i:2d}. {exp_id}: {bitrate:7.2f} Mbps ({reduction:+6.1f}%) - {approach}...")
    
    # Bottom performers
    print(f"\n⚠️  BOTTOM 5 EXPERIMENTS (Highest Bitrate):")
    for i, (exp_id, bitrate, reduction, approach) in enumerate(reversed(performers[-5:]), 1):
        print(f"  ❌ {i}. {exp_id}: {bitrate:7.2f} Mbps ({reduction:+6.1f}%) - {approach}...")

def save_detailed_reports(reports, output_dir='./experiment_postmortems'):
    """Save individual post-mortem reports."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 Saving detailed post-mortem reports to {output_dir}/")
    
    for report in reports:
        exp_id = report['experiment_id']
        filename = f"{output_dir}/{exp_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
    
    # Save summary
    summary_file = f"{output_dir}/SUMMARY.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'total_experiments': len(reports),
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'reports': reports
        }, f, indent=2)
    
    print(f"✅ Saved {len(reports)} detailed reports")
    print(f"📄 Summary saved to: {summary_file}")

def main():
    # Fetch all data
    experiments = fetch_experiments_from_dynamodb()
    code_attempts = fetch_code_attempts_list()
    
    # Build comprehensive reports
    reports = build_comprehensive_report(experiments)
    
    # Print summary
    print_summary_report(reports)
    
    # Save detailed reports
    save_detailed_reports(reports)
    
    print("\n" + "=" * 80)
    print("✅ COMPREHENSIVE POST-MORTEM ANALYSIS COMPLETE!")
    print(f"📁 Detailed reports saved to: ./experiment_postmortems/")
    print(f"📊 Total work logs: {len(reports)}")
    print(f"🗂️  Code attempts available: {len(code_attempts)}")
    print("=" * 80)

if __name__ == '__main__':
    main()

