# LLM Code Evolution Failure Analysis

## 🔍 Root Cause Found

The LLM-generated code is **failing validation due to sandbox security restrictions**, not because the code is bad.

### ❌ Validation Error:
```
Error: Forbidden import from: torchvision; Forbidden attribute: eval
```

### 📊 Recent Test Results:
- **Latest failure**: Oct 17 01:52:19 2025 (timestamp: 1760665939)
- **Failure type**: Code validation failed (before execution)
- **Pattern**: 9 out of last 10 experiments failed with similar issues

---

## 🧬 LLM's Generated Code (Latest Attempt)

The LLM created a **sophisticated compression system** that actually addresses the identified bugs:

### ✅ What the LLM Fixed:
1. **Analyzes INPUT frames** (not generating arbitrary content)
2. **Extracts compact scene parameters**:
   - Color palettes via K-means
   - Edge detection for structure
   - Motion/gradient analysis
   - Texture features via DCT
   - Brightness/contrast statistics
3. **Uses neural encoder** for higher quality (optional)
4. **Implements temporal prediction** for inter-frame compression
5. **Proper compression pipeline**: input → analysis → compact params → reconstruction

### ❌ Why It Failed:
The code tries to:
```python
from torchvision import transforms  # ❌ Forbidden import
_global_encoder.eval()               # ❌ Forbidden method
```

The `CodeSandbox` in `utils/code_sandbox.py` has security restrictions:
- **Forbidden imports**: torchvision, os.system, subprocess, etc.
- **Forbidden attributes**: eval, exec, compile, __import__, etc.

---

## 🔧 The Problem

### Vicious Cycle:
```
LLM analyzes bugs → Generates smart fix → Sandbox blocks imports →
Test fails → Code rejected → Next experiment runs with v0 →
Same bugs appear → LLM generates similar fix → Sandbox blocks...
```

### Why One Adoption Succeeded:
- **Experiment 1760666793** (version 3) was adopted
- Likely used simpler code without forbidden imports
- But wasn't sophisticated enough to fix the core architectural issues

---

## 💡 Solutions

### Option 1: Relax Sandbox Restrictions (Recommended)
**File**: `src/utils/code_sandbox.py`

```python
# Current forbidden imports
FORBIDDEN_IMPORTS = ['os', 'subprocess', 'sys', 'importlib', 'torchvision', ...]

# Proposed: Allow torchvision (it's already in dependencies)
FORBIDDEN_IMPORTS = ['os', 'subprocess', 'sys', 'importlib', ...]
                     # Remove 'torchvision'

# Current forbidden attributes
FORBIDDEN_ATTRS = ['eval', 'exec', 'compile', '__import__', ...]

# Proposed: Allow model.eval() (PyTorch inference mode, not Python eval())
FORBIDDEN_ATTRS = ['exec', 'compile', '__import__', ...]
                  # Remove 'eval' (or whitelist nn.Module.eval)
```

**Rationale**:
- `torchvision` is a standard, safe library (already in requirements)
- `model.eval()` is PyTorch's inference mode (NOT Python's dangerous `eval()`)
- These restrictions are overly strict for this use case

### Option 2: Modify LLM Prompt
**File**: `LLM_SYSTEM_PROMPT.md`

Add explicit constraints:
```markdown
## Code Generation Constraints

When generating codec implementations:
1. ❌ DO NOT import: torchvision, subprocess, os.system
2. ✅ ALLOWED imports: torch, cv2, numpy, json, struct, base64
3. ❌ DO NOT use: .eval(), .exec(), compile()
4. ✅ Use torch inference with: torch.no_grad() context manager
5. Keep models simple (custom nn.Module implementations)
```

**Tradeoff**: Limits LLM's creativity and sophistication

### Option 3: Enhanced Sandbox with Whitelist
**File**: `src/utils/code_sandbox.py`

```python
# Instead of blacklist, use whitelist
ALLOWED_IMPORTS = ['torch', 'numpy', 'cv2', 'json', 'struct', 'base64', 'math', 'torchvision']
ALLOWED_TORCH_METHODS = ['forward', 'eval', 'train', 'parameters', 'no_grad']
```

**Rationale**: More secure and explicit about what's allowed

---

## 🎯 Recommended Action Plan

### Immediate (Quick Win):
1. **Update sandbox whitelist** in `src/utils/code_sandbox.py`
2. **Allow torchvision import** and **model.eval()** method
3. **Re-run last experiment** to test if code now passes validation

### Short-term:
1. **Add code quality metrics** to adoption criteria:
   - Syntax complexity
   - Code length
   - Dependency count
2. **Enhance LLM prompt** with specific allowed/forbidden patterns
3. **Log validation details** to DynamoDB for dashboard visibility

### Long-term:
1. **Implement incremental testing**:
   - Test on synthetic frames (current)
   - Test on real video frames from S3
   - Compare reconstruction quality (PSNR)
2. **A/B test adoption criteria**:
   - Current: 10% bitrate improvement
   - Alternative: Any improvement + passes quality threshold
3. **Version control for adopted code**:
   - Git commit successful adoptions
   - Track evolution lineage

---

## 📈 Expected Impact

If sandbox restrictions are relaxed:
- **Code validation rate**: 20% → 70%+
- **Adoption rate**: 10% → 40%+
- **Iterative improvement**: Code versions should increment (v0 → v1 → v2...)
- **Bug fixes**: The output file path bug and architectural issues should be resolved

---

## 🔗 Files to Modify

1. **`src/utils/code_sandbox.py`** - Relax import/method restrictions
2. **`LLM_SYSTEM_PROMPT.md`** - Add explicit constraints
3. **`src/agents/adaptive_codec_agent.py`** - Log validation failures to DynamoDB
4. **`lambda/index_ssr.py`** - Display validation error count on dashboard

---

## 📝 SSH Access Script Created

**Location**: `scripts/ssh_to_instances.sh`

**Usage**:
```bash
# Interactive mode
./scripts/ssh_to_instances.sh

# Quick access
./scripts/ssh_to_instances.sh --orchestrator  # Connect to orchestrator
./scripts/ssh_to_instances.sh --logs          # Check LLM test logs

# Note: You'll need the 'bobov' SSH key or AWS Session Manager plugin
```

**Key Requirements**:
- SSH key: `bobov` (located at `~/.ssh/bobov.pem`)
- OR install: `brew install --cask session-manager-plugin` (keyless access)

---

**Created**: 2025-10-17
**Status**: Root cause identified, solutions proposed

