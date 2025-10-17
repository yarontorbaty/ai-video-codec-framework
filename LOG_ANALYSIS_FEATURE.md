# Log Analysis Feature - LLM-Powered Failure Diagnosis

**Deployed**: October 17, 2025 04:10 UTC

---

## 🎯 Overview

Added LLM-powered log analysis to automatically diagnose experiment failures and display insights on the dashboard. Each failed experiment now includes:
- **Root cause analysis** from Claude
- **Categorized failure types**  
- **Severity ratings**
- **Actionable fix suggestions**

---

## ✨ What Was Added

### 1. **Log Analyzer** (`src/utils/log_analyzer.py`)
   
**New utility that uses Claude to analyze failure logs.**

**Features**:
- Sends error messages and logs to Claude API
- Parses structured JSON response
- Falls back to pattern matching if API unavailable
- Categorizes failures: `syntax_error`, `import_error`, `validation_error`, `runtime_error`, `timeout`, `resource_error`, `logic_error`
- Assigns severity: `critical`, `high`, `medium`, `low`

**Usage**:
```python
from utils.log_analyzer import LogAnalyzer

analyzer = LogAnalyzer()
analysis = analyzer.analyze_failure(
    experiment_id='test_1234',
    experiment_type='llm_generated_code',
    error_message='Forbidden import from: torchvision',
    logs='Full log output...',
    code_snippet='def my_func()...'  # Optional
)

# Returns:
{
    'failure_category': 'validation_error',
    'root_cause': 'Code validation failed due to security restrictions',
    'fix_suggestion': 'Use only allowed libraries: numpy, cv2, torch',
    'severity': 'high'
}
```

---

### 2. **Enhanced Code Testing** (`src/agents/adaptive_codec_agent.py`)

**Updated `test_generated_code()` method to:**
- Capture all logs during code testing
- Call `LogAnalyzer` when validation fails
- Save analysis to `/tmp/codec_versions/validation_failure_*.txt`
- Store analysis in `self._last_failure_analysis`
- Include analysis in evolution result dict

**Changes**:
```python
# Before
if not is_valid:
    return False, None

# After  
if not is_valid:
    logs = log_capture.getvalue()
    analysis = analyzer.analyze_failure(...)
    
    # Save to file with analysis
    # Store for dashboard
    self._last_failure_analysis = analysis
    
    return False, None
```

**Failure results now include**:
```python
{
    'status': 'test_failed',
    'adopted': False,
    'reason': 'Code testing failed',
    'failure_analysis': {
        'failure_category': '...',
        'root_cause': '...',
        'fix_suggestion': '...',
        'severity': '...'
    }
}
```

---

### 3. **Admin API Updates** (`lambda/admin_api.py`)

**Modified `get_experiments_list()` to extract failure analysis:**

```python
# Extract failure analysis if present
failure_analysis = evolution.get('failure_analysis', {})
if failure_analysis:
    exp_data['failure_analysis'] = {
        'category': failure_analysis.get('failure_category', 'unknown'),
        'root_cause': failure_analysis.get('root_cause', 'N/A'),
        'fix_suggestion': failure_analysis.get('fix_suggestion', 'N/A'),
        'severity': failure_analysis.get('severity', 'unknown')
    }
```

**API response now includes failure_analysis field for failed experiments.**

---

### 4. **Dashboard UI** (`dashboard/admin.js`)

#### New Column: "Analysis"
- Shows severity badge for failed experiments
- Color-coded by severity (critical=red, high=orange, medium=yellow, low=gray)
- Icon varies by category (syntax, import, validation, runtime, timeout, etc.)

#### Interactive Failure Modal
- Click analysis badge to open detailed modal
- Shows:
  - Experiment ID
  - Failure category and severity
  - Root cause explanation
  - Suggested fix
- Beautiful gradient UI with color-coded severity

**Example badges**:
```
🔴 CRITICAL    (syntax_error)
🟠 HIGH        (validation_error)
🟡 MEDIUM      (timeout)
⚪ LOW         (logic_error)
```

**Modal UI**:
```
┌─────────────────────────────────────┐
│ 🐛 Failure Analysis                 │
│ real_exp_1760667123                 │
├─────────────────────────────────────┤
│ Validation Error • HIGH SEVERITY    │
│                                     │
│ 🔍 Root Cause:                      │
│ Code validation failed due to       │
│ security restrictions               │
│                                     │
│ 🔧 Suggested Fix:                   │
│ Use only allowed libraries: numpy,  │
│ cv2, torch, torchvision            │
│                                     │
│               [Close]               │
└─────────────────────────────────────┘
```

---

## 📂 Files Modified/Added

### Added:
- `src/utils/log_analyzer.py` - LLM-powered log analysis

### Modified:
- `src/agents/adaptive_codec_agent.py` - Capture logs, analyze failures
- `lambda/admin_api.py` - Include failure analysis in API response
- `dashboard/admin.js` - Display analysis badges and modal

---

## 🚀 Deployment Status

✅ **Orchestrator**: Deployed (PID: 5691)  
✅ **Admin API Lambda**: Updated (2025-10-17T04:10:45Z)  
✅ **Admin JS (S3)**: Uploaded  
✅ **CloudFront**: Cache invalidated (IAR4US48BG6U4LY2N98XPVC5SB)

---

## 🧪 How It Works

### Flow Diagram:
```
Experiment Runs
     ↓
Code Validation Fails
     ↓
Logs Captured
     ↓
LogAnalyzer.analyze_failure()
     ↓
Claude API Call
     ↓
Structured JSON Response
     ↓
Saved to:
- /tmp/codec_versions/validation_failure_*.txt
- DynamoDB (experiments table)
     ↓
Dashboard Displays:
- Severity badge
- Click → Modal with details
```

---

## 📊 Example Analysis

### Input:
```
Error: Forbidden import from: torchvision; Forbidden attribute: eval
Code: import torch
      import torchvision
      model.eval()
```

### Before This Feature:
```
Status: test_failed
Reason: Code testing failed or produced invalid results
```

### After This Feature:
```
Status: test_failed
Reason: Code testing failed or produced invalid results
Analysis:
  Category: validation_error
  Root Cause: Code validation failed due to forbidden imports (torchvision) 
              and attributes (eval). The sandbox restricts certain libraries 
              for security.
  Fix: Use only allowed libraries: numpy, cv2, torch, json, struct, base64. 
       Replace model.eval() with torch.no_grad() context manager.
  Severity: high
```

---

## 💡 Benefits

### For Debugging:
1. **Instant diagnosis** - No need to manually check logs
2. **Actionable fixes** - Clear suggestions for resolution
3. **Severity assessment** - Prioritize critical failures
4. **Historical tracking** - All analyses saved to disk

### For Dashboard Users:
1. **Visual indicators** - Color-coded severity at a glance
2. **Detailed insights** - Click to see full analysis
3. **Better understanding** - Know why experiments fail
4. **Faster iteration** - Fix issues based on suggestions

### For LLM Evolution:
1. **Better prompts** - Can feed failure analysis back to planning LLM
2. **Self-improvement** - System learns from failure patterns
3. **Reduced repetition** - Avoid same mistakes

---

## 🔮 Future Enhancements

### Phase 2:
- [ ] **Aggregate statistics**: Track failure categories over time
- [ ] **Trending issues**: Dashboard chart of common failures
- [ ] **Auto-fix suggestions**: Generate PR with fixes
- [ ] **Email alerts**: Notify on critical failures

### Phase 3:
- [ ] **Failure prediction**: Warn before likely failures
- [ ] **Code review**: Analyze code BEFORE running
- [ ] **Historical comparison**: "Similar failure in exp #42"
- [ ] **LLM feedback loop**: Use failure analysis in next hypothesis

---

## 🧰 Testing

### Test the Analyzer:
```python
cd src/utils
python3 log_analyzer.py
```

### Test on Next Experiment:
1. Wait for autonomous orchestrator to run next experiment
2. If it fails, check:
   - `/tmp/codec_versions/validation_failure_*.txt` (on EC2)
   - Dashboard → Experiments table → Analysis column
   - Click badge → See modal with details

---

## 📝 Configuration

### LLM Model:
- Uses `claude-sonnet-4-20250514`
- Max tokens: 1024
- Timeout: 30 seconds

### Fallback Behavior:
If Claude API fails, uses pattern matching:
- `forbidden import` → validation_error, high severity
- `syntaxerror` → syntax_error, high severity
- `timeout` → timeout, medium severity
- `not better than` → logic_error, low severity

---

## 🔐 Security

### API Key:
- Stored in AWS Secrets Manager: `ai-video-codec/anthropic-api-key`
- Retrieved securely by LogAnalyzer
- Not logged or exposed

### Sandbox Isolation:
- Log analysis happens AFTER sandbox execution
- No additional security risks
- Logs are truncated to last 3000 chars before sending to API

---

## 📞 Troubleshooting

### Analysis not showing on dashboard?
1. **Check experiment has failure_analysis field**:
   ```bash
   aws dynamodb scan --table-name ai-video-codec-experiments \
     --region us-east-1 | jq '.Items[-1].experiments.S' | jq
   ```

2. **Check API response**:
   Open browser DevTools → Network → `/admin/experiments` → Check response

3. **Hard refresh browser**: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows)

### Claude API errors?
- Check logs: `./scripts/ssh_to_instances.sh --logs`
- Verify API key: AWS Secrets Manager console
- Fallback will still provide basic analysis

---

**Status**: ✅ Fully deployed and operational  
**Next Test**: Watch for next experiment failure (within ~10 minutes)


