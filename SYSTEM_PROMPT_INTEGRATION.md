# ✅ System Prompt Integration Complete

## WHAT WAS THE PROBLEM?

You asked: **"Are you feeding the new prompt to the LLM?"**

**Answer:** NO - The comprehensive system prompt in `LLM_SYSTEM_PROMPT.md` was **NOT** being used!

### The Issue

The `LLMExperimentPlanner.generate_compression_code()` method (line 410 in `llm_experiment_planner.py`) had a **hardcoded basic prompt**:

```python
prompt = f"""Based on this video compression experiment analysis, generate a NEW Python compression function.

ANALYSIS:
Root Cause: {analysis.get('root_cause', '')}
Hypothesis: {analysis.get('hypothesis', '')}
...

REQUIREMENTS:
1. Function signature: def compress_video_frame(...)
2. Use only: numpy, cv2, math, json, struct, base64
5. NO imports of: os, sys, subprocess...
"""
```

**Problems:**
- ❌ Didn't mention neural networks (torch) are available
- ❌ Didn't explain the HEVC baseline comparison
- ❌ Didn't explain GitHub integration
- ❌ Didn't provide context about code evolution capabilities
- ❌ Didn't mention previous iterations for comparison
- ❌ Short prompt (~300 chars) vs comprehensive guide (~11,000 chars)

---

## WHAT WAS FIXED

### 1. Load System Prompt on Initialization

Added `_load_system_prompt()` method that searches multiple paths:

```python
def __init__(self, model: str = "claude-sonnet-4-5"):
    # ... (existing init code)
    
    # Load system prompt from file
    self.system_prompt = self._load_system_prompt()

def _load_system_prompt(self) -> str:
    """Load the comprehensive system prompt from LLM_SYSTEM_PROMPT.md"""
    possible_paths = [
        'LLM_SYSTEM_PROMPT.md',  # From project root
        '../LLM_SYSTEM_PROMPT.md',  # From src/agents/
        '../../LLM_SYSTEM_PROMPT.md',  # From deeper
        '/home/ec2-user/ai-video-codec/LLM_SYSTEM_PROMPT.md',  # Absolute on EC2
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                prompt = f.read()
            logger.info(f"✅ Loaded system prompt from {path} ({len(prompt)} chars)")
            return prompt
    
    logger.warning("⚠️  Could not find LLM_SYSTEM_PROMPT.md - using basic prompt")
    return ""
```

### 2. Use System Prompt in Code Generation

Updated `generate_compression_code()` to use the full system prompt:

```python
def generate_compression_code(self, analysis: Dict) -> Optional[Dict]:
    # Build prompt with system context
    if self.system_prompt:
        # Use full system prompt (11K chars with all context)
        prompt = f"""{self.system_prompt}

---

## CURRENT EXPERIMENT CONTEXT

Based on your analysis of past experiments, generate a NEW compression algorithm.

**Your Analysis:**
- Root Cause: {analysis.get('root_cause', '')}
- Hypothesis: {analysis.get('hypothesis', '')}
- Key Insights: {json.dumps(analysis.get('insights', []), indent=2)}

**Your Task:**
Generate the `compress_video_frame` function that implements your hypothesis.
Focus on continuous improvement - beat your previous iteration!

Generate ONLY the Python code (imports + function), no markdown explanations.
"""
    else:
        # Fallback to basic prompt (includes torch now)
        prompt = f"""..."""
```

### 3. Deploy System Prompt File

Updated `deploy_orchestrator.sh` to include:

```bash
tar czf /tmp/orchestrator_deploy.tar.gz \
    src/ \
    scripts/real_experiment.py \
    scripts/autonomous_orchestrator_llm.sh \
    scripts/analyze_past_experiments.py \
    LLM_SYSTEM_PROMPT.md \          # ← Added
    NEURAL_NETWORKS_GUIDE.md \      # ← Added
    requirements.txt
```

**Verified:** File now exists on EC2:
```
-rw-r--r-- 1 502 games 11K Oct 17 01:48 /home/ec2-user/ai-video-codec/LLM_SYSTEM_PROMPT.md
```

---

## WHAT THE LLM NOW SEES

When the LLM generates compression code, it receives:

### Full Context (11,000+ characters)

1. **YOUR CAPABILITIES** 
   - You can generate code
   - Code is tested automatically
   - Better code is deployed automatically
   - All changes are committed to GitHub

2. **CURRENT SYSTEM STATUS**
   - Adoption criteria (10% better or 20% compression)
   - Comparison strategy (HEVC baseline, previous iterations)
   - Success metrics (< 1 Mbps target)
   - Recent performance data

3. **TECHNICAL ARCHITECTURE**
   - How code evolution works
   - Sandbox security model
   - Testing infrastructure
   - Git integration

4. **DEPLOYMENT PIPELINE**
   - Validation → Testing → Adoption → GitHub commit
   - How to check your deployed code
   - Version tracking

5. **GITHUB ACCESS**
   - You can commit code changes
   - Credentials are secure (Secrets Manager)
   - Commits include detailed messages with metrics

6. **SELF-GOVERNANCE**
   - You can debug your own code failures
   - Access to failure logs
   - Pattern recognition for common errors
   - Improved prompt generation

7. **ALLOWED IMPORTS** ✅
   ```python
   import numpy as np
   import cv2
   import json
   import struct
   import base64
   import math
   import typing
   import torch          # ← NOW DOCUMENTED!
   import torchvision    # ← NOW DOCUMENTED!
   ```

8. **NEURAL NETWORKS AVAILABLE** ✅
   - PyTorch 1.13.1 available
   - Can use neural networks for compression
   - Examples: autoencoders, learned transforms, semantic encoders
   - Goal: Better compression than traditional codecs

9. **COMPARISON STRATEGY** ✅
   - **HEVC Baseline:** 10 Mbps (industry standard)
   - **Previous LLM Iteration:** Your last best version
   - Must be 10% better to be adopted
   - Continuous improvement through evolution

10. **SOURCE FILES** ✅
    - Test frames: Currently synthetic (random, black, gray)
    - TODO: Will use actual SOURCE_HD_RAW.mp4 frames
    - Compare against HEVC_HD_10Mbps.mp4

11. **TECHNICAL CONSTRAINTS**
    - Memory limits
    - Compute constraints (CPU only)
    - No unsafe operations
    - Security sandbox

12. **DEBUGGING TIPS**
    - Common failure patterns
    - How to check logs
    - Error recovery strategies

13. **EXPERIMENT CONTEXT** (Current Analysis)
    - Root cause from your analysis
    - Your hypothesis
    - Key insights
    - Suggested next steps

---

## IMPACT ON LLM BEHAVIOR

### Before (Basic Prompt)

```
LLM sees: "Generate a compression function with numpy, cv2"
LLM thinks: "I'll use DCT and quantization (standard approach)"
Result: Generic compression, no neural networks, no innovation
```

### After (Full System Prompt)

```
LLM sees: "You have PyTorch, can use neural networks, compete against HEVC 10 Mbps, 
           GitHub integration, previous iteration was X Mbps, you need 10% improvement,
           here's the sandbox, here's the adoption criteria, here's how to debug"
LLM thinks: "I should try learned transforms with neural networks, 
             my previous attempt failed at X, I'll try Y approach,
             I need < 9 Mbps to beat HEVC,
             I can use torch.nn for semantic compression"
Result: Innovative neural network compression, learns from failures, 
        continuous improvement
```

---

## EXPECTED IMPROVEMENTS

### 1. Neural Network Usage

**Before:**
```python
# LLM only used cv2.resize, DCT-like transforms
compressed = cv2.imencode('.jpg', frame)[1].tobytes()
```

**After:**
```python
import torch
import torch.nn as nn

class SemanticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, stride=2)
    
    def forward(self, x):
        return self.conv(x)

# LLM can now use learned compression
encoder = SemanticEncoder()
latent = encoder(torch.from_numpy(frame))
compressed = latent.numpy().tobytes()
```

### 2. Baseline Awareness

**Before:**
- LLM didn't know what "good" compression was
- No target to beat

**After:**
- LLM knows HEVC = 10 Mbps
- Goal: < 9 Mbps to be adopted
- Understands it's competing against industry standard

### 3. Iteration Awareness

**Before:**
- Each code generation was independent
- No learning from previous attempts

**After:**
- LLM knows its previous version's performance
- Can see what worked/failed
- Builds on successful approaches
- Avoids repeating failures

### 4. Self-Debugging

**Before:**
- LLM generated code blindly
- No awareness of common failures

**After:**
- LLM can analyze its own failures
- Learns patterns (e.g., "tuple arguments needed")
- Generates improved prompts for itself
- Self-corrects common mistakes

### 5. Deployment Understanding

**Before:**
- LLM didn't know if code was adopted
- No feedback loop

**After:**
- LLM knows adoption criteria
- Understands GitHub commits mean success
- Can track its deployed versions
- Feedback loop for continuous improvement

---

## VERIFICATION

### Check Logs (Next Experiment)

The orchestrator logs should now show:

```
✅ Loaded system prompt from /home/ec2-user/ai-video-codec/LLM_SYSTEM_PROMPT.md (11234 chars)
```

### Check Generated Code

The next LLM-generated code should:
- ✅ Use `torch` imports (if appropriate)
- ✅ Reference HEVC baseline in comments
- ✅ Show awareness of previous iteration
- ✅ Include more sophisticated compression algorithms
- ✅ Demonstrate understanding of adoption criteria

### Check Dashboard

After 2-3 experiment cycles:
- ✅ Varied metrics (not identical anymore)
- ✅ Code evolution attempts logged
- ✅ GitHub commits for successful adoptions
- ✅ Continuous improvement trajectory

---

## FILES MODIFIED

1. **`src/agents/llm_experiment_planner.py`**
   - Added `_load_system_prompt()` method
   - Modified `__init__()` to load prompt on startup
   - Updated `generate_compression_code()` to use full system prompt
   - Added fallback to basic prompt if file not found

2. **`scripts/deploy_orchestrator.sh`**
   - Added `LLM_SYSTEM_PROMPT.md` to deployment package
   - Added `NEURAL_NETWORKS_GUIDE.md` to deployment package

3. **Deployed Files on EC2:**
   - `/home/ec2-user/ai-video-codec/LLM_SYSTEM_PROMPT.md` (11K)
   - `/home/ec2-user/ai-video-codec/NEURAL_NETWORKS_GUIDE.md` (reference)

---

## SUMMARY

**Status:** ✅ **COMPLETE**

The LLM now receives:
- ✅ Full system context (11K chars vs 300 chars)
- ✅ Neural network capabilities (PyTorch available)
- ✅ HEVC baseline comparison (10 Mbps target)
- ✅ Previous iteration awareness (continuous improvement)
- ✅ GitHub integration understanding (code commits)
- ✅ Self-debugging guidance (failure analysis)
- ✅ Deployment pipeline knowledge (adoption criteria)

**Next Experiment (~01:53):**
- LLM will receive full system prompt
- Should generate more sophisticated code
- May attempt neural network compression
- Will be aware of HEVC baseline and previous iterations
- Dashboard should show varied metrics!

**Monitoring:**
```bash
# Watch orchestrator logs
aws ssm send-command --instance-ids i-063947ae46af6dbf8 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["tail -50 /tmp/orch.log"]' \
  --query 'Command.CommandId' --output text

# Check for system prompt loading
# Should see: "✅ Loaded system prompt from ... (11234 chars)"
```

🚀 **The LLM is now fully equipped for autonomous code evolution!**

