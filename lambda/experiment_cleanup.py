#!/usr/bin/env python3
"""
Experiment Cleanup Lambda
Runs every 5 minutes to detect and close out stuck/abandoned experiments.
"""

import json
import boto3
import time
from datetime import datetime
from decimal import Decimal

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
experiments_table = dynamodb.Table('ai-video-codec-experiments')

# Timeout thresholds (in seconds)
# With retries and quality verification, experiments can take much longer
PHASE_TIMEOUTS = {
    'design': 600,              # 10 minutes (LLM analysis)
    'deploy': 120,              # 2 minutes (code deployment)
    'validation': 3600,         # 60 minutes (validation with up to 10 retries)
    'execution': 7200,          # 120 minutes (execution with up to 10 retries + quality check)
    'quality_verification': 1800,  # 30 minutes (decompression + PSNR/SSIM calculation)
    'analysis': 300,            # 5 minutes (store results + blog update)
}

# Maximum overall experiment time (6 hours)
# Only timeout if actually stuck (no progress)
MAX_EXPERIMENT_TIME = 21600  # 6 hours

# Minimum time before checking for stuck status (1 hour)
# Don't timeout experiments younger than this
MIN_EXPERIMENT_AGE = 3600


def handler(event, context):
    """
    Cleanup handler triggered by CloudWatch Events every 5 minutes.
    """
    print("🧹 Starting experiment cleanup...")
    
    try:
        # Scan for running experiments
        response = experiments_table.scan(
            FilterExpression='#status = :running',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={':running': 'running'}
        )
        
        running_experiments = response.get('Items', [])
        
        if not running_experiments:
            print("  ✅ No running experiments found")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'No cleanup needed'})
            }
        
        print(f"  Found {len(running_experiments)} running experiment(s)")
        
        current_time = int(time.time())
        cleaned_up = []
        
        for exp in running_experiments:
            exp_id = exp.get('experiment_id', 'unknown')
            start_time = exp.get('start_time')
            timestamp = exp.get('timestamp', 0)
            current_phase = exp.get('current_phase', 'unknown')
            elapsed_seconds = exp.get('elapsed_seconds', 0)
            validation_retries = exp.get('validation_retries', 0)
            execution_retries = exp.get('execution_retries', 0)
            
            # Determine experiment age
            if start_time:
                age = current_time - int(float(start_time))
            else:
                age = current_time - int(timestamp)
            
            # Check if experiment is making progress
            # If retries are increasing or elapsed_seconds is updating, it's working
            has_recent_activity = (elapsed_seconds > 0 and age - elapsed_seconds < 600)  # Activity within last 10 min
            has_retries = (validation_retries > 0 or execution_retries > 0)  # Actively retrying
            
            # Check if experiment is stuck
            is_stuck = False
            reason = None
            
            # SKIP: Don't timeout young experiments (< 1 hour)
            if age < MIN_EXPERIMENT_AGE:
                print(f"  ⏳ {exp_id}: Too young to timeout ({age}s < {MIN_EXPERIMENT_AGE}s)")
                continue
            
            # Check 1: Overall timeout (6 hours) - but only if no recent activity
            if age > MAX_EXPERIMENT_TIME:
                if has_recent_activity or has_retries:
                    print(f"  ⚠️  {exp_id}: Old but still active (retries: val={validation_retries}, exec={execution_retries})")
                else:
                    is_stuck = True
                    reason = f"Exceeded maximum experiment time ({MAX_EXPERIMENT_TIME}s = {MAX_EXPERIMENT_TIME//3600}h) with no recent activity"
            
            # Check 2: Phase-specific timeout - but only if no progress
            elif current_phase in PHASE_TIMEOUTS:
                phase_timeout = PHASE_TIMEOUTS[current_phase]
                if age > phase_timeout:
                    # Check if making progress (retries)
                    if has_retries and (validation_retries < 10 or execution_retries < 10):
                        print(f"  🔄 {exp_id}: In {current_phase} with retries (val={validation_retries}, exec={execution_retries})")
                    elif has_recent_activity:
                        print(f"  🔄 {exp_id}: In {current_phase} with recent activity (elapsed: {elapsed_seconds}s)")
                    else:
                        is_stuck = True
                        reason = f"Stuck in {current_phase} phase (>{phase_timeout}s = {phase_timeout//60} min) with no progress"
            
            # Check 3: Unknown phase for too long (no progress indicator)
            elif current_phase == 'unknown' and age > 600:
                is_stuck = True
                reason = f"Unknown phase for {age}s (>{600}s = 10 min) - likely crashed"
            
            if is_stuck:
                print(f"  🔴 {exp_id}: {reason}")
                cleanup_experiment(exp_id, exp, reason, age)
                cleaned_up.append(exp_id)
            else:
                if age > 3600:  # Log if running over 1 hour
                    print(f"  ✅ {exp_id}: Running normally (age: {age}s = {age//3600}h {(age%3600)//60}m, phase: {current_phase})")
                else:
                    print(f"  ✅ {exp_id}: Running normally (age: {age}s, phase: {current_phase})")
        
        result = {
            'cleaned_up_count': len(cleaned_up),
            'cleaned_up_ids': cleaned_up,
            'total_running': len(running_experiments)
        }
        
        print(f"🧹 Cleanup complete: {len(cleaned_up)} experiment(s) closed out")
        
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
        
    except Exception as e:
        print(f"❌ Cleanup error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def cleanup_experiment(exp_id: str, exp: dict, reason: str, age: int):
    """
    Close out a stuck experiment and update blog.
    
    Args:
        exp_id: Experiment ID
        exp: Experiment data from DynamoDB
        reason: Why it was cleaned up
        age: How long it's been running (seconds)
    """
    from decimal import Decimal
    
    def convert_floats(obj):
        if isinstance(obj, dict):
            return {k: convert_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_floats(item) for item in obj]
        elif isinstance(obj, float):
            return Decimal(str(obj))
        return obj
    
    try:
        # Get existing data
        current_phase = exp.get('current_phase', 'unknown')
        timestamp = exp.get('timestamp', int(time.time()))
        timestamp_iso = exp.get('timestamp_iso', datetime.utcnow().isoformat() + 'Z')
        
        # Create experiments array with failure info
        experiments_array = [{
            'experiment_type': 'real_procedural_generation',
            'status': 'timed_out',
            'real_metrics': {},
            'comparison': {},
            'failure_reason': reason,
            'stuck_phase': current_phase,
            'runtime_seconds': age,
            'abandoned': True
        }]
        
        # Update experiment with timeout status
        updated_data = {
            'experiment_id': exp_id,
            'timestamp': timestamp,  # Preserve original
            'timestamp_iso': timestamp_iso,
            'status': 'timed_out',
            'experiments': json.dumps(experiments_array),
            'phase_completed': current_phase,
            'needs_human': True,
            'human_intervention_reasons': [
                f"Experiment abandoned: {reason}",
                f"Stuck in {current_phase} phase after {age}s ({age//60} minutes)",
                "System may have crashed or hung - check orchestrator logs"
            ],
            'cleanup_timestamp': int(time.time()),
            'cleaned_up_by': 'automated_cleanup'
        }
        
        experiments_table.put_item(Item=convert_floats(updated_data))
        print(f"  ✅ {exp_id}: Marked as timed_out and blog updated")
        
    except Exception as e:
        print(f"  ❌ {exp_id}: Failed to cleanup - {e}")


def test_locally():
    """Test function locally"""
    result = handler({}, {})
    print("\n" + "="*60)
    print("TEST RESULT:")
    print("="*60)
    print(json.dumps(json.loads(result['body']), indent=2))


if __name__ == '__main__':
    test_locally()

