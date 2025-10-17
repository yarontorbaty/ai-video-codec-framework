# ✅ OPTION C: Simplified LLM-Only Experiments

## WHAT CHANGED

### Before (3 Sub-Experiments):
1. **LLM-Generated Code** - Test new code (was failing)
2. **Procedural Generation** - Baseline using parameter storage (identical 0.04 Mbps results)
3. **Neural Networks Test** - Just health check (no compression)

### After (1 Focused Experiment):
1. **LLM Autonomous Code Evolution** - ONLY experiment that matters
   - Compares against HEVC baseline (10 Mbps)
   - Compares against previous LLM iteration
   - Uses neural networks as tools (PyTorch available)
   - Produces varied results based on algorithm quality

---

## HOW IT WORKS NOW

### Experiment Flow

```
┌─────────────────────────────────────┐
│  1. LLM Pre-Analysis                │
│     - Analyze past 20 experiments   │
│     - Generate hypothesis           │
│     - Plan improvement strategy     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  2. LLM Generates Compression Code  │
│     - Python function               │
│     - Can use: numpy, cv2, torch    │
│     - Returns compressed bytes      │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  3. Code Validation                 │
│     - AST security checks           │
│     - No unsafe operations          │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  4. Code Execution & Testing        │
│     - Test on 3 sample frames       │
│     - Calculate compression metrics │
│     - Measure bitrate               │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  5. Comparison Decision             │
│                                     │
│  First Iteration:                   │
│    Compare vs HEVC (10 Mbps)        │
│    Adopt if < 9 Mbps OR < 15 Mbps   │
│                                     │
│  Subsequent Iterations:              │
│    Compare vs Previous Best         │
│    Adopt if 10% better bitrate      │
│    OR 20% better compression        │
└────────────┬────────────────────────┘
             │
             ▼
      ┌──────┴──────┐
      │             │
    ADOPT        REJECT
      │             │
      ▼             ▼
┌───────────┐   ┌────────────┐
│  Deploy   │   │  Log &     │
│  + GitHub │   │  Continue  │
│  Commit   │   │  Iterating │
└───────────┘   └────────────┘
```

---

## COMPARISON STRATEGY

### Baseline 1: HEVC (Industry Standard)

**File:** `s3://ai-video-codec-videos-580473065386/hevc/HEVC_HD_10Mbps.mp4`
- **Size:** 12.7 MB
- **Bitrate:** 10 Mbps
- **Quality:** 1080p @ 30fps
- **Codec:** H.265/HEVC

**Target:** Beat HEVC (< 9 Mbps) on first successful iteration

### Baseline 2: Previous LLM Iteration

- Each new version competes against the last adopted version
- Tracks incremental improvement
- Version 1 → Version 2 → Version 3 → ...
- Must show 10% improvement to be adopted

### Source Video (Future)

**File:** `s3://ai-video-codec-videos-580473065386/source/SOURCE_HD_RAW.mp4`
- **Size:** 744 MB (raw/uncompressed)
- **Current:** Using synthetic test frames (fast iteration)
- **TODO:** Load actual frames from source video for testing

---

## NEURAL NETWORKS INTEGRATION

### Available to LLM

The LLM can now use **PyTorch** in its generated code:

```python
import torch
import torch.nn as nn
import torchvision

def compress_video_frame(frame, frame_index, config):
    # Example: Use neural network for semantic compression
    encoder = SemanticEncoder()  # LLM can define this
    latent = encoder(torch.from_numpy(frame))
    compressed = entropy_encode(latent)
    return compressed
```

### Neural Networks Are Tools, Not Separate Experiments

- **Before:** Neural networks were tested separately (just a health check)
- **After:** LLM can use them as part of compression algorithm
- **Example Use Cases:**
  - Learned transform coding (replace DCT)
  - Semantic feature extraction
  - Motion prediction networks
  - Generative decoders
  - Learned entropy models

---

## WHY THIS FIXES THE DASHBOARD

### The Problem

Dashboard showed identical metrics (0.04 Mbps) because:
1. LLM code test was **failing** (no new metrics)
2. Procedural generation **succeeded** with same baseline (0.04 Mbps)
3. Dashboard displayed experiment #2 → Same results every time

### The Solution

Now:
1. **Only LLM code runs** → No fallback to baseline
2. Each LLM iteration produces **different algorithms**
3. Different algorithms → **Different compression ratios**
4. Dashboard shows **varied metrics** as LLM improves

### Expected Dashboard Behavior

After deployment:
- **First few experiments:** May show 15-20 Mbps (not better than HEVC yet)
- **Learning phase:** LLM iterates, metrics improve over time
- **Breakthrough:** Eventually < 10 Mbps (beats HEVC)
- **Optimization:** Continues improving toward < 1 Mbps goal
- **Commits to GitHub:** Each adoption creates a new commit with metrics

---

## ADOPTION CRITERIA LOGIC

### First Iteration (No History)

```python
if bitrate < 9.0:  # 10% better than HEVC
    return ADOPT
elif bitrate < 15.0 and code_works:
    return ADOPT  # Give it a chance to iterate
else:
    return REJECT
```

### Subsequent Iterations

```python
if new_bitrate < prev_bitrate * 0.9:  # 10% better
    return ADOPT
elif new_compression > prev_compression * 1.2:  # 20% better
    return ADOPT
else:
    return REJECT  # Keep iterating
```

---

## WHAT'S IN EACH EXPERIMENT NOW

### DynamoDB Structure

```json
{
  "experiment_id": "real_exp_1760664995",
  "timestamp": "2025-10-17T01:36:35.028",
  "experiments": [
    {
      "experiment_type": "llm_generated_code_evolution",
      "status": "completed",
      "evolution": {
        "status": "adopted" | "rejected" | "test_failed",
        "adopted": true | false,
        "version": 1,
        "metrics": {
          "bitrate_mbps": 8.5,
          "compression_ratio": 4.2
        },
        "improvement": "15.3% better than v0",
        "summary": "LLM evolved codec to v1 - 8.5 Mbps, 4.2x compression",
        "github_committed": true,
        "github_commit_hash": "abc123..."
      }
    }
  ]
}
```

### What's Removed

- ❌ `real_procedural_generation` experiment
- ❌ `real_ai_neural_networks` experiment
- ✅ Only `llm_generated_code_evolution` remains

---

## FILE CHANGES

### Modified Files

1. **`scripts/real_experiment.py`**
   - Removed procedural generation experiment
   - Removed neural networks test
   - Kept only LLM code evolution
   - Added baseline comparison logging

2. **`src/agents/adaptive_codec_agent.py`**
   - Updated `should_adopt_new_code()` with HEVC baseline comparison
   - First iteration compares vs 10 Mbps HEVC
   - Subsequent iterations compare vs previous best
   - Better logging of comparison decisions

3. **`LLM_SYSTEM_PROMPT.md`**
   - Added PyTorch to allowed imports
   - Explained neural networks are available as tools
   - Added comparison strategy section
   - Clarified HEVC baseline and iteration comparison

### Fixed Bugs

1. **Argument Format Bug**
   - Was: `execute_function(code, name, frame, i, config)` ❌
   - Now: `execute_function(code, name, (frame, i, config))` ✅
   - Tupled arguments correctly

2. **Parameter Bug**
   - Was: Missing `frame_index` parameter ❌
   - Now: Passes `(frame, frame_index, config)` ✅

---

## EXPECTED TIMELINE

### Immediate (Next 2-5 Minutes)

- LLM generates new code
- Code validates and executes successfully ✅
- Metrics calculated (bitrate, compression)
- May not be adopted yet (not 10% better)
- **Dashboard shows NEW metrics** (not 0.04 anymore!)

### Short Term (Next Hour)

- Multiple iterations run
- LLM learns from failures
- Code quality improves
- Eventually: First successful adoption
- GitHub commit created
- Dashboard shows "Code Evolved to v1"

### Medium Term (Next Few Hours)

- Continuous improvement cycle
- v1 → v2 → v3 → ...
- Each version 10% better than previous
- Dashboard tracks evolution progress
- GitHub history shows all commits

### Long Term Goal

- Beat HEVC efficiency (< 10 Mbps)
- Approach 1 Mbps target
- Potentially use neural networks for state-of-art compression
- Compete with learned codecs like VVC/AV1

---

## MONITORING

### Check Current Status

```bash
# See latest experiment
aws dynamodb scan --table-name ai-video-codec-experiments --limit 1 --region us-east-1

# Check orchestrator logs
aws ssm send-command --instance-ids i-063947ae46af6dbf8 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["tail -50 /tmp/orch.log"]' \
  --query 'Command.CommandId' --output text

# Check for code adoption
aws ssm send-command --instance-ids i-063947ae46af6dbf8 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["cat /tmp/best_codec_implementation.json 2>&1 || echo \"No adoption yet\""]' \
  --query 'Command.CommandId' --output text

# Check GitHub commits
git fetch && git log origin/master --oneline -10
```

### Dashboard Should Show

- ✅ Varied metrics (not identical anymore)
- ✅ Code evolution status (badges)
- ✅ Version numbers (v1, v2, v3...)
- ✅ GitHub commit links
- ✅ Improvement percentages

---

## SUMMARY

**Option C Implementation:**
- ✅ Simplified from 3 experiments to 1 focused experiment
- ✅ LLM-only evolution (no separate baselines)
- ✅ Compares against HEVC (10 Mbps) and previous iterations
- ✅ Neural networks available as compression tools (PyTorch)
- ✅ Fixed argument passing bugs
- ✅ Ready for autonomous code evolution

**Next:** Wait 2-3 minutes for the next experiment cycle to see varied metrics on the dashboard!

🚀 **True autonomous AI research system - learning, evolving, committing!**

