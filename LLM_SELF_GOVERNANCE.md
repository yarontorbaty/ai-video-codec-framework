# LLM Self-Governance System

## Overview

The AI Video Codec research system now includes **complete LLM self-governance** - the ability for the LLM to debug itself, understand its failures, generate fixes, and continuously improve without human intervention.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│         LLM Code Generation (Every 60 seconds)          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              LLM Self-Debugger                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  1. Analyze previous failures (last hour)        │   │
│  │  2. Identify patterns (forbidden imports, etc.)  │   │
│  │  3. Generate recommendations                     │   │
│  │  4. Create improved prompt                       │   │
│  │  5. Produce self-governance report               │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Enhanced CodeSandbox                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │  1. Save code attempt to /tmp/code_attempts/     │   │
│  │  2. Validate code (AST analysis)                 │   │
│  │  3. Execute with safety constraints              │   │
│  │  4. Save detailed error logs if fails            │   │
│  │  5. Return success/failure with metrics          │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │ Validation Failed       │ Execution Failed
        ▼                         ▼
┌──────────────────┐     ┌───────────────────┐
│ Save to:         │     │ Save to:          │
│ - attempt_*.py   │     │ - error_*.txt     │
│ - validation_    │     │ - Full traceback  │
│   failure_*.txt  │     │ - Code + context  │
└──────────────────┘     └───────────────────┘
        │                         │
        └────────────┬────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Next Iteration (with improved prompt)            │
│         Cycle continues until success!                   │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. Enhanced CodeSandbox (`src/utils/code_sandbox.py`)

**New Features:**
- Saves every code validation attempt to `/tmp/code_attempts/attempt_{timestamp}.py`
- Logs validation failures with line numbers and code excerpts
- Saves execution errors with full tracebacks to `error_{timestamp}.txt`
- Verbose logging of exactly WHY code fails

**Example Output:**
```
❌ VALIDATION FAILED: Forbidden import: os
Code excerpt:
import os
import numpy as np

def compress_video_frame(frame, config):
...
```

### 2. Improved Adaptive Codec Agent (`src/agents/adaptive_codec_agent.py`)

**New Features:**
- Saves ALL code versions to `/tmp/codec_versions/codec_attempt_{timestamp}.py`
- Creates validation failure logs with full context
- Tracks every attempt (pass or fail)
- Logs first 500 chars of failing code for analysis

**Version Tracking:**
```
/tmp/codec_versions/
├── codec_attempt_1760654527.py
├── codec_attempt_1760654528.py
├── codec_attempt_1760654529.py ✅ (ADOPTED as v1)
└── validation_failure_1760654527.txt
```

### 3. LLM Self-Debugger (`src/agents/llm_self_debugger.py`)

**Core Capabilities:**

#### A. Failure Analysis
```python
debugger = LLMSelfDebugger()
analysis = debugger.analyze_recent_failures(lookback_hours=1)
```

Identifies patterns:
- `forbidden_imports` - LLM using disallowed modules
- `syntax_errors` - Code with syntax issues
- `missing_function` - Required function not defined
- `timeouts` - Code taking too long
- `other` - Other execution errors

#### B. Recommendation Generation

For each pattern, generates:
- **Issue**: What's wrong
- **Count**: How many times it occurred
- **Description**: Detailed explanation
- **Fix**: Specific action to take
- **Severity**: critical/high/medium/low
- **Auto-fixable**: Whether it can be fixed automatically

**Example Recommendation:**
```json
{
  "issue": "forbidden_imports",
  "count": 5,
  "description": "LLM is trying to import forbidden modules: os, sys",
  "fix": "Update LLM prompt to only use: numpy, cv2, torch, math, json, struct, base64, io",
  "severity": "high",
  "auto_fixable": true
}
```

#### C. Improved Prompt Generation

Based on failures, generates enhanced prompts:

```python
improved_prompt = debugger.generate_improved_prompt(current_failures)
```

Includes:
- Base requirements (function signature, imports, constraints)
- Specific fixes for detected issues
- Examples of correct code structure
- Performance requirements

#### D. Self-Governance Reports

Comprehensive reports for LLM review:

```python
report = debugger.create_self_governance_report()
```

Contains:
- System status (code evolution working?, success rate)
- Failure analysis (patterns, recommendations)
- Improved prompt (ready to use)
- Action items (prioritized, with status)

### 4. Experiment Integration (`scripts/real_experiment.py`)

**Self-Governance Flow:**

Every experiment:
1. Run `LLMSelfDebugger.analyze_recent_failures()`
2. If > 5 failures detected:
   - Generate self-governance report
   - Log recommendations
   - Use improved prompt for next generation
3. Proceed with code generation and testing
4. Results feed back into debugger for next cycle

**Example Log Output:**
```
⚠️  Detected 7 recent failures
🔧 Generating self-governance report...
   💡 LLM is trying to import forbidden modules: Update prompt to only use allowed modules
   💡 Generated code does not define function: Add function signature requirement
```

## File Structure

### Debugging Artifacts

```
/tmp/
├── code_attempts/                  # All validation/execution attempts
│   ├── attempt_1760654527.py       # Code that was tested
│   ├── attempt_1760654528.py
│   ├── error_1760654527.txt        # Execution error with traceback
│   └── error_1760654528.txt
│
├── codec_versions/                 # All code versions
│   ├── codec_attempt_1760654527.py # Every generation attempt
│   ├── codec_attempt_1760654528.py
│   ├── codec_attempt_1760654529.py # ✅ Working version
│   ├── validation_failure_1760654527.txt
│   └── validation_failure_1760654528.txt
│
├── best_codec_implementation.json  # Current best (when one works)
│
└── self_governance_report_*.json   # Self-analysis reports
```

## Usage

### For Debugging

**View recent attempts:**
```bash
aws ssm send-command \
  --instance-ids i-063947ae46af6dbf8 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["ls -lt /tmp/code_attempts/ | head -10"]'
```

**View latest error:**
```bash
aws ssm send-command \
  --instance-ids i-063947ae46af6dbf8 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["cat /tmp/code_attempts/error_*.txt | tail -50"]'
```

**View codec versions:**
```bash
aws ssm send-command \
  --instance-ids i-063947ae46af6dbf8 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["ls -lt /tmp/codec_versions/"]'
```

**Check self-governance in logs:**
```bash
aws ssm send-command \
  --instance-ids i-063947ae46af6dbf8 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["grep -i \"debug\\|governance\\|recommendation\" /tmp/orch.log | tail -20"]'
```

### For LLM Chat Interface

The LLM can query its own governance status:

**Via Admin Chat:**
```
You: "Analyze why code generation is failing"

LLM: [Reads self-governance reports]
     [Analyzes /tmp/code_attempts/]
     [Reviews failure patterns]
     
     Response: "I've analyzed the failures. The main issue is:
     1. Forbidden imports (5 occurrences) - I was trying to use 'os' module
     2. Missing function (3 occurrences) - Not defining compress_video_frame
     
     I'm now using an improved prompt that:
     - Only imports allowed modules
     - Explicitly defines the required function
     - Includes performance constraints
     
     Next attempt should pass validation."
```

## Self-Governance Cycle

### Example Evolution Sequence

**Iteration 1:**
```
LLM generates:
  import os  # ❌ Forbidden!
  def my_compress(data):  # ❌ Wrong function name!
    ...

Result: ❌ VALIDATION FAILED: Forbidden import: os
Debugger: Detects forbidden_imports: 1, missing_function: 1
Recommendation: "Only use numpy, cv2; Define compress_video_frame"
```

**Iteration 2:**
```
LLM generates (with improved prompt):
  import numpy as np
  def compress_video_frame(frame, config):  # ✅ Correct!
    return frame  # ❌ Wrong return format!
    
Result: ❌ EXECUTION FAILED: Expected dict, got ndarray
Debugger: Detects return_format_error: 1
Recommendation: "Must return dict with 'compressed' and 'metadata' keys"
```

**Iteration 3:**
```
LLM generates (further improved):
  import numpy as np
  import cv2
  
  def compress_video_frame(frame, config):  # ✅ Correct!
    quality = int(config.get('quality', 0.8) * 100)
    _, compressed = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return {
      'compressed': compressed.tobytes(),
      'metadata': {'size': len(compressed)}
    }  # ✅ Correct format!
    
Result: ✅ VALIDATION PASSED ✅ EXECUTION PASSED
Performance: 3.2 Mbps, 2.5x compression
Status: 🎉 ADOPTED as codec v1!
```

**Iteration 4:**
```
LLM generates improved version:
  # Better algorithm...
  
Result: ✅ ALL TESTS PASSED
Performance: 2.1 Mbps (34% better than v1!)
Status: 🎉 ADOPTED as codec v2!
```

## Benefits

### 1. Complete Transparency
- Every code attempt is saved
- Every failure is logged with full details
- LLM can review its own history
- Human can inspect all artifacts

### 2. Self-Correction
- LLM identifies its own mistakes
- Generates fixes automatically
- Improves prompts based on failures
- Learns from each iteration

### 3. Rapid Iteration
- No human intervention needed
- System runs continuously
- Failures analyzed in real-time
- Improvements applied immediately

### 4. Governance
- LLM controls the entire process
- Decides when to retry
- Chooses which fixes to apply
- Tracks its own progress

## Monitoring

### Key Metrics

- **Success Rate**: % of code generations that pass validation
- **Failure Patterns**: Types and frequencies of failures
- **Time to First Success**: How long until first working code
- **Evolution Progress**: Number of versions adopted

### Dashboard Integration

The health monitoring dashboard shows:
- Code generation success/failure rate
- Most recent failures and recommendations
- Evolution timeline (v1, v2, v3...)
- LLM self-governance status

## Troubleshooting

### No Code Being Generated

Check if LLM is enabled:
```bash
aws secretsmanager get-secret-value \
  --secret-id ai-video-codec/anthropic-api-key \
  --region us-east-1
```

### All Code Failing Validation

Review self-governance report:
```bash
cat /tmp/self_governance_report_*.json | tail -n 1 | jq .
```

Check recommendations and ensure improved prompt is being used.

### Code Passes Validation But Fails Execution

Check execution errors:
```bash
ls -lt /tmp/code_attempts/error_*.txt | head -1 | xargs cat
```

The error log will show the exact traceback and issue.

## Future Enhancements

1. **Multi-LLM Collaboration**: Multiple LLMs working together
2. **Evolutionary Algorithms**: Combine best features from multiple versions
3. **Automated Testing**: Generate comprehensive test suites
4. **Performance Prediction**: Predict if code will work before testing
5. **Code Review**: LLM reviews other LLM's code

## Conclusion

The system now has **complete self-governance**:
- ✅ Sees every attempt
- ✅ Understands every failure
- ✅ Generates fixes automatically
- ✅ Improves continuously
- ✅ Tracks progress
- ✅ Governs itself

This is **true autonomous AI research** with self-correction and continuous improvement!

