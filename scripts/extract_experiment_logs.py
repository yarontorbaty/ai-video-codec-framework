#!/usr/bin/env python3
"""
Extract and analyze per-experiment logs for post-mortem analysis.

This script:
1. Fetches the orchestrator log from EC2
2. Parses it to extract per-experiment sections
3. Analyzes each experiment's journey (design → deploy → validate → execute → analyze)
4. Generates detailed post-mortem reports
"""

import boto3
import re
import json
from datetime import datetime
from collections import defaultdict

def fetch_orchestrator_log():
    """Fetch orchestrator log from EC2 via SSM."""
    print("📥 Fetching orchestrator log from EC2...")
    
    ec2 = boto3.client('ec2', region_name='us-east-1')
    ssm = boto3.client('ssm', region_name='us-east-1')
    
    # Get orchestrator instance
    response = ec2.describe_instances(
        Filters=[
            {'Name': 'tag:Name', 'Values': ['ai-video-codec-orchestrator']},
            {'Name': 'instance-state-name', 'Values': ['running']}
        ]
    )
    
    instance_id = response['Reservations'][0]['Instances'][0]['InstanceId']
    
    # Send command to get log
    cmd_response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': ['cat /tmp/orch.log']}
    )
    
    command_id = cmd_response['Command']['CommandId']
    
    # Wait for command to complete
    import time
    for _ in range(10):
        time.sleep(1)
        result = ssm.get_command_invocation(
            CommandId=command_id,
            InstanceId=instance_id
        )
        if result['Status'] in ['Success', 'Failed']:
            break
    
    if result['Status'] != 'Success':
        raise Exception(f"Command failed: {result['Status']}")
    
    log_content = result['StandardOutputContent']
    print(f"✅ Fetched {len(log_content)} bytes")
    
    return log_content

def parse_experiments(log_content):
    """Parse log content to extract per-experiment sections."""
    print("\n🔍 Parsing experiments from log...")
    
    experiments = {}
    current_exp = None
    current_lines = []
    
    for line in log_content.split('\n'):
        # Check for experiment ID
        exp_match = re.search(r'ID: (proc_exp_\d+)', line)
        if exp_match:
            # Save previous experiment
            if current_exp and current_lines:
                experiments[current_exp] = '\n'.join(current_lines)
            
            # Start new experiment
            current_exp = exp_match.group(1)
            current_lines = [line]
        elif current_exp:
            current_lines.append(line)
    
    # Save last experiment
    if current_exp and current_lines:
        experiments[current_exp] = '\n'.join(current_lines)
    
    print(f"✅ Found {len(experiments)} experiments in log")
    return experiments

def analyze_experiment(exp_id, log_text):
    """Analyze a single experiment's log."""
    analysis = {
        'experiment_id': exp_id,
        'phases': {},
        'errors': [],
        'warnings': [],
        'metrics': {},
        'timeline': []
    }
    
    # Extract phases
    phase_patterns = {
        'design': r'PHASE 1: DESIGN',
        'deploy': r'PHASE 2: DEPLOY',
        'validation': r'PHASE 3: VALIDATION',
        'execution': r'PHASE 4: EXECUTION',
        'analysis': r'PHASE 5: ANALYSIS'
    }
    
    for phase, pattern in phase_patterns.items():
        if re.search(pattern, log_text):
            analysis['phases'][phase] = 'started'
            
            # Check for completion
            if re.search(rf'{pattern}.*✅', log_text, re.DOTALL):
                analysis['phases'][phase] = 'completed'
            elif re.search(rf'{pattern}.*❌', log_text, re.DOTALL):
                analysis['phases'][phase] = 'failed'
    
    # Extract errors
    error_lines = re.findall(r'ERROR:.*', log_text)
    analysis['errors'] = error_lines[:10]  # Limit to first 10
    
    # Extract warnings
    warning_lines = re.findall(r'WARNING:.*', log_text)
    analysis['warnings'] = warning_lines[:10]
    
    # Extract metrics
    bitrate_match = re.search(r'Bitrate: ([\d.]+) Mbps', log_text)
    if bitrate_match:
        analysis['metrics']['bitrate_mbps'] = float(bitrate_match.group(1))
    
    reduction_match = re.search(r'Reduction: ([-\d.]+)%', log_text)
    if reduction_match:
        analysis['metrics']['reduction_percent'] = float(reduction_match.group(1))
    
    # Extract code generation info
    code_match = re.search(r'Code generated: (\d+) characters', log_text)
    if code_match:
        analysis['metrics']['code_chars'] = int(code_match.group(1))
    
    # Extract retry counts
    val_retry_match = re.search(r'Validation attempt (\d+)/(\d+)', log_text)
    if val_retry_match:
        analysis['metrics']['validation_attempts'] = int(val_retry_match.group(1))
        analysis['metrics']['max_validation_retries'] = int(val_retry_match.group(2))
    
    exec_retry_match = re.search(r'Execution attempt (\d+)/(\d+)', log_text)
    if exec_retry_match:
        analysis['metrics']['execution_attempts'] = int(exec_retry_match.group(1))
        analysis['metrics']['max_execution_retries'] = int(exec_retry_match.group(2))
    
    # Extract hypothesis
    hypo_match = re.search(r'Hypothesis: (.+)', log_text)
    if hypo_match:
        analysis['hypothesis'] = hypo_match.group(1)[:200]  # First 200 chars
    
    return analysis

def generate_report(experiments_analysis):
    """Generate post-mortem report."""
    print("\n" + "=" * 80)
    print("📊 POST-MORTEM ANALYSIS REPORT")
    print("=" * 80)
    
    # Summary statistics
    total = len(experiments_analysis)
    completed = sum(1 for a in experiments_analysis.values() if a['phases'].get('analysis') == 'completed')
    failed = sum(1 for a in experiments_analysis.values() if any(p == 'failed' for p in a['phases'].values()))
    
    print(f"\n📈 SUMMARY:")
    print(f"  Total experiments: {total}")
    print(f"  Completed: {completed} ({completed/total*100:.1f}%)")
    print(f"  Failed: {failed} ({failed/total*100:.1f}%)")
    
    # Phase completion rates
    print(f"\n📊 PHASE COMPLETION RATES:")
    phases = ['design', 'deploy', 'validation', 'execution', 'analysis']
    for phase in phases:
        completed_phase = sum(1 for a in experiments_analysis.values() if a['phases'].get(phase) == 'completed')
        print(f"  {phase.capitalize()}: {completed_phase}/{total} ({completed_phase/total*100:.1f}%)")
    
    # Common errors
    print(f"\n❌ COMMON ERRORS:")
    error_counts = defaultdict(int)
    for analysis in experiments_analysis.values():
        for error in analysis['errors']:
            # Extract error type
            error_type = error.split(':')[1].split('(')[0].strip() if ':' in error else error[:50]
            error_counts[error_type] += 1
    
    for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {count}x: {error[:70]}")
    
    # Bitrate distribution
    print(f"\n📈 BITRATE DISTRIBUTION:")
    bitrates = [a['metrics'].get('bitrate_mbps', 0) for a in experiments_analysis.values() if a['metrics'].get('bitrate_mbps')]
    if bitrates:
        bitrates.sort()
        print(f"  Min: {min(bitrates):.2f} Mbps")
        print(f"  Median: {bitrates[len(bitrates)//2]:.2f} Mbps")
        print(f"  Max: {max(bitrates):.2f} Mbps")
        print(f"  Average: {sum(bitrates)/len(bitrates):.2f} Mbps")
        
        # Histogram
        print(f"\n  Distribution:")
        ranges = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 50), (50, 100)]
        for low, high in ranges:
            count = sum(1 for b in bitrates if low <= b < high)
            bar = '█' * (count * 50 // len(bitrates))
            print(f"  {low:3d}-{high:3d} Mbps: {bar} {count}")
    
    # Retry statistics
    print(f"\n🔄 RETRY STATISTICS:")
    val_retries = [a['metrics'].get('validation_attempts', 1) - 1 for a in experiments_analysis.values() if 'validation_attempts' in a['metrics']]
    exec_retries = [a['metrics'].get('execution_attempts', 1) - 1 for a in experiments_analysis.values() if 'execution_attempts' in a['metrics']]
    
    if val_retries:
        print(f"  Validation retries: avg={sum(val_retries)/len(val_retries):.1f}, max={max(val_retries)}")
    if exec_retries:
        print(f"  Execution retries: avg={sum(exec_retries)/len(exec_retries):.1f}, max={max(exec_retries)}")
    
    # Top performers
    print(f"\n🏆 TOP 5 PERFORMERS:")
    performers = [(exp_id, a['metrics'].get('bitrate_mbps', 999)) 
                  for exp_id, a in experiments_analysis.items() 
                  if a['metrics'].get('bitrate_mbps')]
    performers.sort(key=lambda x: x[1])
    
    for i, (exp_id, bitrate) in enumerate(performers[:5], 1):
        reduction = experiments_analysis[exp_id]['metrics'].get('reduction_percent', 0)
        print(f"  {i}. {exp_id}: {bitrate:.2f} Mbps ({reduction:.1f}% reduction)")
    
    return experiments_analysis

def save_detailed_logs(experiments_analysis, output_dir='./experiment_logs'):
    """Save detailed per-experiment logs."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 Saving detailed logs to {output_dir}/")
    
    for exp_id, analysis in experiments_analysis.items():
        filename = f"{output_dir}/{exp_id}_analysis.json"
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    print(f"✅ Saved {len(experiments_analysis)} detailed logs")

def main():
    # Fetch log
    log_content = fetch_orchestrator_log()
    
    # Parse experiments
    experiments_logs = parse_experiments(log_content)
    
    # Analyze each experiment
    print("\n🔬 Analyzing experiments...")
    experiments_analysis = {}
    for exp_id, log_text in experiments_logs.items():
        experiments_analysis[exp_id] = analyze_experiment(exp_id, log_text)
    
    # Generate report
    generate_report(experiments_analysis)
    
    # Save detailed logs
    save_detailed_logs(experiments_analysis)
    
    print("\n✅ Post-mortem analysis complete!")
    print(f"📁 Detailed logs saved to: ./experiment_logs/")

if __name__ == '__main__':
    main()

