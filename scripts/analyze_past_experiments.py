#!/usr/bin/env python3
"""
One-time script to analyze all past experiments with LLM
"""
import boto3
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.llm_experiment_planner import LLMExperimentPlanner

def main():
    print("🔍 Analyzing all past experiments with LLM...")
    
    # Initialize DynamoDB
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    experiments_table = dynamodb.Table('ai-video-codec-experiments')
    reasoning_table = dynamodb.Table('ai-video-codec-reasoning')
    
    # Fetch all experiments
    response = experiments_table.scan()
    experiments = response.get('Items', [])
    
    print(f"📊 Found {len(experiments)} experiments")
    
    # Check which experiments already have reasoning
    reasoning_response = reasoning_table.scan()
    existing_reasoning_exp_ids = set([r.get('experiment_id') for r in reasoning_response.get('Items', [])])
    
    # Initialize LLM planner
    planner = LLMExperimentPlanner()
    
    analyzed_count = 0
    skipped_count = 0
    
    # Sort experiments by timestamp
    experiments.sort(key=lambda x: x.get('timestamp', 0))
    
    for exp in experiments:
        exp_id = exp.get('experiment_id', '')
        
        # Skip if already analyzed
        if exp_id in existing_reasoning_exp_ids:
            print(f"  ⏭️  {exp_id[:30]}... (already analyzed)")
            skipped_count += 1
            continue
        
        print(f"  🤖 Analyzing {exp_id}...")
        
        try:
            # Parse experiment data
            experiments_data = json.loads(exp.get('experiments', '[]'))
            
            # Create experiment summary for LLM
            summary = {
                'experiment_id': exp_id,
                'timestamp': exp.get('timestamp', 0),
                'status': exp.get('status', 'unknown'),
                'experiments': experiments_data
            }
            
            # Get past experiments (all experiments before this one)
            past_experiments = [e for e in experiments if e.get('timestamp', 0) < exp.get('timestamp', 0)]
            
            # Analyze with LLM
            reasoning = planner.analyze_experiment(summary, past_experiments)
            
            # Store reasoning in DynamoDB
            if reasoning:
                reasoning_table.put_item(Item=reasoning)
                print(f"     ✅ Analysis stored")
                analyzed_count += 1
            else:
                print(f"     ⚠️  No reasoning generated")
                
        except Exception as e:
            print(f"     ❌ Error: {e}")
    
    print(f"\n📈 Summary:")
    print(f"   Analyzed: {analyzed_count}")
    print(f"   Skipped (already done): {skipped_count}")
    print(f"   Total: {len(experiments)}")
    print("\n✅ Past experiment analysis complete!")

if __name__ == '__main__':
    main()

