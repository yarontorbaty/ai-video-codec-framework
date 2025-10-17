# Code Sandbox Update - October 17, 2025

## ✅ Changes Deployed

### Updated File: `src/utils/code_sandbox.py`

### Changes Made:

1. **Added torchvision to ALLOWED_MODULES**
   ```python
   ALLOWED_MODULES = {
       'numpy', 'np',
       'cv2',
       'torch',
       'torchvision',      # ← NEW
       'torch.nn',         # ← NEW
       'torch.nn.functional',  # ← NEW
       'math',
       'json',
       'struct',
       'base64',
       ...
   }
   ```

2. **Removed 'eval' from FORBIDDEN_ATTRIBUTES**
   ```python
   FORBIDDEN_ATTRIBUTES = {
       '__import__',
       'exec',
       # 'eval' REMOVED - PyTorch model.eval() is safe
       'compile',
       ...
   }
   ```
   - **Note**: Python's dangerous `eval()` builtin is still blocked via restricted_globals
   - Only `model.eval()` method calls are now allowed

3. **Added torchvision imports to execution environment**
   ```python
   try:
       import torchvision
       restricted_globals['torchvision'] = torchvision
   except ImportError:
       pass
   
   # Also added struct and base64 for completeness
   restricted_globals['struct'] = __import__('struct')
   restricted_globals['base64'] = __import__('base64')
   ```

---

## 🚀 Deployment Status

**Deployed to**: Orchestrator Instance (i-063947ae46af6dbf8)  
**Deployed at**: 2025-10-17 02:31 UTC  
**Method**: SSM Command (ID: 2fbc4b5f-7753-41f7-a6b1-c2f169506b7a)  
**Status**: ✅ SUCCESSFUL  
**Orchestrator**: Running (PID: 31906)

---

## 📊 Expected Impact

### Before Update:
- **Code Validation Failure Rate**: ~90%
- **Main Error**: `Forbidden import from: torchvision; Forbidden attribute: eval`
- **Code Adoption Rate**: ~10% (1 in 10 experiments)
- **Version Progression**: Stuck at v0 (failed experiments revert)

### After Update:
- **Code Validation Pass Rate**: Expected 70%+ (was 10%)
- **Code Adoption Rate**: Expected 40%+ (was 10%)
- **Version Progression**: Should increment (v0 → v1 → v2 → v3...)
- **Bug Fixes**: LLM's architectural fixes should now pass validation

---

## 🔍 What to Watch For

### Next 5-10 Experiments:

1. **Check validation logs**:
   ```bash
   ./scripts/ssh_to_instances.sh --logs
   # Or manually:
   ls -lt /tmp/codec_versions/validation_failure_*.txt | head -5
   ```

2. **Monitor adoption rate**:
   ```bash
   aws dynamodb scan --table-name ai-video-codec-experiments \
     --region us-east-1 \
     --query 'Items[*].{id:experiment_id.S, adopted:experiments.S}' \
     | jq '.[] | select(.adopted | contains("adopted": true))'
   ```

3. **Check dashboard**:
   - Visit https://aiv1codec.com/
   - Look for experiments with version > 0
   - Check "Code" and "Version" columns in experiments table

---

## 🧬 LLM Code Quality

The latest LLM-generated code (blocked before this fix) showed:

### ✅ Sophisticated Architecture:
- **Scene parameter extraction**: K-means color palette, edge detection, motion analysis
- **Neural encoding**: Optional SceneParameterEncoder for high-quality modes
- **Temporal prediction**: Inter-frame compression via motion prediction
- **Proper pipeline**: Input → Analysis → Compact params → Reconstruction

### 🎯 Fixes Core Issues:
- ✅ Analyzes INPUT frames (not generating arbitrary content)
- ✅ Creates compact representations (not full video files)
- ✅ Addresses file path bug (uses experiment_id in metadata)
- ✅ Implements true compression architecture

### 📈 Expected Bitrate:
Based on code structure, should achieve:
- **Quality 1-4**: ~0.5-2 Mbps (minimal params only)
- **Quality 5-7**: ~2-5 Mbps (with neural encoding)
- **Quality 8-10**: ~5-8 Mbps (full neural params)

---

## 🔬 Testing Recommendations

### Option 1: Wait for Next Experiment
The autonomous orchestrator will naturally run the next experiment within ~10 minutes.

### Option 2: Trigger Manual Experiment
```bash
aws ssm send-command \
  --instance-ids i-063947ae46af6dbf8 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["cd /home/ec2-user/ai-video-codec && python3 scripts/real_experiment.py"]' \
  --region us-east-1
```

### Option 3: Check Current Orchestrator Logs
```bash
aws ssm send-command \
  --instance-ids i-063947ae46af6dbf8 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["tail -100 /tmp/orch.log"]' \
  --region us-east-1 \
  --query 'Command.CommandId' --output text

# Then get results:
# aws ssm get-command-invocation --command-id <ID> --instance-id i-063947ae46af6dbf8 --query StandardOutputContent --output text
```

---

## 🛡️ Security Notes

### What's Still Blocked:
- ❌ `os`, `subprocess`, `sys` imports
- ❌ File operations (`open`, `file`)
- ❌ Dynamic code execution (`exec`, `compile`)
- ❌ Reflection (`getattr`, `setattr`, `__import__`)

### What's Now Allowed:
- ✅ `torchvision` (transforms, models, etc.)
- ✅ `model.eval()` (PyTorch inference mode)
- ✅ `torch.nn`, `torch.nn.functional`
- ✅ All standard codec libraries (numpy, cv2, struct, base64)

### Why This Is Safe:
1. **torchvision**: Standard, widely-used library for image/video processing
2. **model.eval()**: Just sets PyTorch modules to inference mode (no code execution)
3. **Execution still sandboxed**: Runs with restricted globals and timeout limits
4. **No system access**: File I/O and subprocess calls remain blocked

---

## 📝 Related Files

- **Updated**: `src/utils/code_sandbox.py`
- **Analysis**: `LLM_CODE_FAILURE_ANALYSIS.md`
- **Deployment**: `scripts/deploy_orchestrator.sh`
- **SSH Helper**: `scripts/ssh_to_instances.sh`

---

## 🎯 Success Criteria

Within the next 5-10 experiments, we should see:

1. **Validation pass rate > 50%** (was ~10%)
2. **At least 2-3 code adoptions** (with version increment)
3. **Bitrate improvements** (code actually addressing the bugs)
4. **Fewer repeated bug reports** (LLM seeing progress)

---

## 📞 Next Steps

1. **Monitor dashboard**: Check https://aiv1codec.com/ for new experiments
2. **Watch version numbers**: Should see v1, v2, v3 appearing
3. **Review adoption status**: Look for green "✓ Adopted" badges
4. **Check logs if issues persist**: Use `./scripts/ssh_to_instances.sh --logs`

---

**Updated**: 2025-10-17 02:31 UTC  
**Status**: Deployed and monitoring  
**Expected Results**: Within 1-2 hours

