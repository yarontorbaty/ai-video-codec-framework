#!/usr/bin/env python3
"""
LLM Self-Debugger
Analyzes code generation failures and provides fixes to the LLM.
Enables autonomous self-governance and improvement.
"""

import os
import json
import logging
import boto3
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMSelfDebugger:
    """
    Analyzes code generation failures and enables the LLM to self-correct.
    """
    
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        self.control_table = self.dynamodb.Table('ai-video-codec-control')
        
    def analyze_recent_failures(self, lookback_hours: int = 1) -> Dict:
        """
        Analyze recent code generation failures and identify patterns.
        """
        logger.info(f"🔍 Analyzing failures from last {lookback_hours} hours...")
        
        # Check /tmp/code_attempts for failures
        attempt_dir = '/tmp/code_attempts'
        if not os.path.exists(attempt_dir):
            return {
                'total_attempts': 0,
                'total_failures': 0,
                'failure_patterns': [],
                'recommendation': 'No attempts found yet'
            }
        
        # Collect error files
        error_files = [f for f in os.listdir(attempt_dir) if f.startswith('error_')]
        validation_failures = [f for f in os.listdir('/tmp/codec_versions') 
                              if f.startswith('validation_failure_')]
        
        # Analyze errors
        failure_patterns = {}
        all_errors = []
        
        for error_file in error_files:
            try:
                with open(os.path.join(attempt_dir, error_file), 'r') as f:
                    content = f.read()
                    all_errors.append(content)
                    
                    # Extract error type
                    if 'Forbidden import' in content:
                        failure_patterns['forbidden_imports'] = failure_patterns.get('forbidden_imports', 0) + 1
                    elif 'Syntax error' in content:
                        failure_patterns['syntax_errors'] = failure_patterns.get('syntax_errors', 0) + 1
                    elif 'not found in code' in content:
                        failure_patterns['missing_function'] = failure_patterns.get('missing_function', 0) + 1
                    elif 'Timeout' in content:
                        failure_patterns['timeouts'] = failure_patterns.get('timeouts', 0) + 1
                    else:
                        failure_patterns['other'] = failure_patterns.get('other', 0) + 1
            except Exception as e:
                logger.warning(f"Could not read error file {error_file}: {e}")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(failure_patterns, all_errors)
        
        analysis = {
            'timestamp': datetime.utcnow().isoformat(),
            'lookback_hours': lookback_hours,
            'total_attempts': len(os.listdir(attempt_dir)),
            'total_failures': len(error_files) + len(validation_failures),
            'failure_patterns': failure_patterns,
            'recommendations': recommendations,
            'sample_errors': all_errors[:3] if all_errors else []
        }
        
        logger.info(f"📊 Analysis complete: {len(error_files)} errors, {len(failure_patterns)} patterns")
        
        # Save analysis to DynamoDB
        self._save_analysis(analysis)
        
        return analysis
    
    def _generate_recommendations(self, patterns: Dict, all_errors: List[str]) -> List[Dict]:
        """Generate recommendations based on failure patterns."""
        recommendations = []
        
        # Recommendation 1: Forbidden imports
        if patterns.get('forbidden_imports', 0) > 0:
            forbidden_modules = set()
            for error in all_errors:
                if 'Forbidden import' in error:
                    # Extract module name
                    try:
                        module = error.split('Forbidden import')[1].split(':')[1].strip().split()[0]
                        forbidden_modules.add(module)
                    except:
                        pass
            
            recommendations.append({
                'issue': 'forbidden_imports',
                'count': patterns['forbidden_imports'],
                'description': f'LLM is trying to import forbidden modules: {", ".join(forbidden_modules)}',
                'fix': f'Update LLM prompt to only use: numpy, cv2, torch, math, json, struct, base64, io',
                'severity': 'high',
                'auto_fixable': True
            })
        
        # Recommendation 2: Syntax errors
        if patterns.get('syntax_errors', 0) > 0:
            recommendations.append({
                'issue': 'syntax_errors',
                'count': patterns['syntax_errors'],
                'description': 'LLM is generating code with syntax errors',
                'fix': 'Improve LLM prompt with: "Generate syntactically valid Python 3.7+ code"',
                'severity': 'high',
                'auto_fixable': True
            })
        
        # Recommendation 3: Missing function
        if patterns.get('missing_function', 0) > 0:
            recommendations.append({
                'issue': 'missing_function',
                'count': patterns['missing_function'],
                'description': 'Generated code does not define the required function',
                'fix': 'Update prompt: "Must define function compress_video_frame(frame, config)"',
                'severity': 'critical',
                'auto_fixable': True
            })
        
        # Recommendation 4: Timeouts
        if patterns.get('timeouts', 0) > 0:
            recommendations.append({
                'issue': 'timeouts',
                'count': patterns['timeouts'],
                'description': 'Generated code is taking too long to execute',
                'fix': 'Add to prompt: "Code must execute in < 1 second per frame"',
                'severity': 'medium',
                'auto_fixable': True
            })
        
        return recommendations
    
    def _save_analysis(self, analysis: Dict):
        """Save analysis to DynamoDB control table."""
        try:
            import time
            self.control_table.put_item(
                Item={
                    'control_id': f"debug_analysis_{int(time.time())}",
                    'timestamp': int(time.time()),
                    'type': 'debug_analysis',
                    'analysis': json.dumps(analysis)
                }
            )
            logger.info("💾 Saved analysis to DynamoDB")
        except Exception as e:
            logger.warning(f"Could not save analysis: {e}")
    
    def generate_improved_prompt(self, current_failures: Dict) -> str:
        """
        Generate an improved prompt based on current failures.
        This is what the LLM should use for next attempt.
        """
        base_prompt = """
Generate a Python function for video frame compression.

Requirements:
- Function name MUST be: compress_video_frame(frame, config)
- frame: numpy array (H, W, 3) uint8 RGB image
- config: dict with 'quality' parameter (0.0 to 1.0)
- Return: dict with 'compressed' (bytes) and 'metadata' (dict)

Allowed imports ONLY:
- import numpy as np
- import cv2
- import math
- import json
- import struct
- import base64

Code constraints:
- Pure Python 3.7+ syntax only
- No async code
- Must execute in < 1 second per frame
- No file I/O operations
- No network operations

Example structure:
```python
import numpy as np
import cv2

def compress_video_frame(frame, config):
    quality = config.get('quality', 0.8)
    
    # Your compression logic here
    # ...
    
    compressed_data = b'...'  # Your compressed bytes
    
    return {
        'compressed': compressed_data,
        'metadata': {'size': len(compressed_data)}
    }
```
"""
        
        # Add specific fixes based on failures
        if current_failures:
            base_prompt += "\n\n⚠️  CRITICAL FIXES NEEDED:\n"
            
            for rec in current_failures.get('recommendations', []):
                if rec['severity'] in ['critical', 'high']:
                    base_prompt += f"\n- {rec['description']}: {rec['fix']}"
        
        return base_prompt
    
    def create_self_governance_report(self) -> Dict:
        """
        Create a comprehensive report for the LLM to self-govern.
        """
        analysis = self.analyze_recent_failures(lookback_hours=2)
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_status': {
                'code_evolution_working': analysis['total_failures'] == 0,
                'total_attempts': analysis['total_attempts'],
                'success_rate': 0 if analysis['total_attempts'] == 0 else 
                               (1 - analysis['total_failures'] / analysis['total_attempts']) * 100
            },
            'failure_analysis': analysis,
            'improved_prompt': self.generate_improved_prompt(analysis),
            'action_items': [
                {
                    'action': 'update_generation_prompt',
                    'priority': 'high',
                    'description': 'Use improved prompt for next code generation',
                    'status': 'ready'
                },
                {
                    'action': 'review_failed_attempts',
                    'priority': 'medium',
                    'description': f'Review {analysis["total_failures"]} failed attempts in /tmp/code_attempts',
                    'status': 'pending'
                },
                {
                    'action': 'test_simple_baseline',
                    'priority': 'high',
                    'description': 'Start with simplest possible compression (JPEG encode)',
                    'status': 'ready'
                }
            ]
        }
        
        # Save report
        try:
            import time
            report_file = f'/tmp/self_governance_report_{int(time.time())}.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📄 Created self-governance report: {report_file}")
        except Exception as e:
            logger.warning(f"Could not save report: {e}")
        
        return report


def test_debugger():
    """Test the self-debugger."""
    debugger = LLMSelfDebugger()
    
    print("Running self-diagnostic...")
    analysis = debugger.analyze_recent_failures(lookback_hours=24)
    
    print("\n" + "="*60)
    print("FAILURE ANALYSIS")
    print("="*60)
    print(f"Total Attempts: {analysis['total_attempts']}")
    print(f"Total Failures: {analysis['total_failures']}")
    print(f"\nFailure Patterns:")
    for pattern, count in analysis['failure_patterns'].items():
        print(f"  - {pattern}: {count}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"\n{i}. {rec['description']}")
        print(f"   Fix: {rec['fix']}")
        print(f"   Severity: {rec['severity']}")
    
    print("\n" + "="*60)
    print("IMPROVED PROMPT")
    print("="*60)
    print(debugger.generate_improved_prompt(analysis))
    
    print("\n" + "="*60)
    print("SELF-GOVERNANCE REPORT")
    print("="*60)
    report = debugger.create_self_governance_report()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    test_debugger()

