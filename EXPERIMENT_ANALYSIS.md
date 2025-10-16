# AI Codec Experiment Analysis

## Issues Found and Fixed

### 1. ✅ Bitrate Display Confusion (FIXED)

**Problem:**
- All experiments showed `-50.4%` or `+50.4%` which was unclear
- Negative values looked like improvements but were actually **regressions**

**Root Cause:**
The compression calculation was correct, but the display was confusing:
- **Negative** = file is LARGER than baseline (BAD)
- **Positive** = file is SMALLER than baseline (GOOD)

**Fix:**
Updated dashboard to show clear indicators:
- `↓ X%` in green = Good (file got smaller)
- `↑ X%` in red = Bad (file got larger)

**Current Status:**
All experiments now show **↑ 50.4%** in red, indicating the compressed files are **50% LARGER** than the HEVC baseline (10 Mbps → 15 Mbps).

---

### 2. ⚠️ Identical Results Across All Experiments

**Problem:**
All 9 experiments have identical metrics:
- Bitrate: 15.038184 Mbps
- Compression: -50.38% (50% worse than baseline)
- Status: All completed successfully

**Root Cause - THIS IS AN EXPERIMENT ISSUE:**

The procedural generation code is **not actually varying between experiments**. Looking at `src/agents/procedural_generator.py`:

```python
def generate_procedural_video(self, output_path, duration=10.0, fps=30.0):
    # ALWAYS uses the same parameters
    # No variation based on past experiments
    # No learning from LLM suggestions
```

**Why This Happens:**
1. Each experiment runs the **exact same code** with **exact same parameters**
2. The LLM reasoning is stored in DynamoDB but **not fed back into the experiment**
3. The agents don't read past LLM suggestions or adapt their approach
4. Result: Same input → Same output every time

**What Should Happen:**
1. LLM analyzes past failures
2. LLM suggests parameter changes (e.g., "try higher compression ratio", "use different codec")
3. **Next experiment reads LLM suggestions and applies them**
4. Results improve (or fail in a new way)
5. Repeat

---

### 3. ✅ LLM Analysis Integration (FIXED)

**Problem:**
- LLM reasoning was generated but not integrated into experiment workflow
- No pre-experiment hypothesis generation
- No post-experiment analysis

**Fix:**
Updated `scripts/real_experiment.py` to include:

**Before Experiment:**
1. Fetch all past experiments
2. LLM generates hypothesis: "Based on past failures, try X approach"
3. Hypothesis is logged and available for the agent

**After Experiment:**
1. LLM analyzes results: "Root cause of failure was Y"
2. LLM generates next experiment plan
3. Analysis is stored in DynamoDB for dashboard/blog display

**New Script:**
Created `scripts/analyze_past_experiments.py` to run one-time analysis on all 9 past experiments.

---

## Current AI Codec Performance

**Baseline (HEVC):** 10 Mbps  
**Current AI Codec:** 15.04 Mbps  
**Performance:** ↑ 50.4% (WORSE) ❌

**Why It's Failing:**
The procedural generation approach creates mathematical descriptions of scenes, but:
1. The descriptions themselves are too large
2. No adaptive compression based on scene complexity
3. No hybrid approach using both procedural + neural compression
4. Parameters are not optimized

---

## Next Steps

### Immediate Actions:

1. **Run Past Analysis:**
   ```bash
   python scripts/analyze_past_experiments.py
   ```
   This will generate LLM reasoning for all 9 past experiments.

2. **Make Agents Read LLM Suggestions:**
   Update `procedural_generator.py` to:
   - Accept LLM suggestions as parameters
   - Vary compression approach based on suggestions
   - Try different mathematical functions based on past failures

3. **Monitor New Experiments:**
   The orchestrator will now:
   - Generate pre-experiment hypotheses
   - Run experiments
   - Analyze results post-experiment
   - Each experiment should be DIFFERENT

### Expected Timeline:

- **Next 6 hours:** LLM analysis appears on dashboard/blog for all past experiments
- **Next 24 hours:** New experiments start using LLM suggestions
- **Next 48 hours:** Results should show variation (some better, some worse, but all DIFFERENT)
- **Next 7 days:** Gradual improvement towards < 1 Mbps target

---

## How to Verify It's Working

1. **Check Dashboard:**
   - Each experiment should have different methods
   - Bitrate should vary (not all 15.04 Mbps)
   - Blog posts should show unique LLM reasoning

2. **Check DynamoDB:**
   ```bash
   aws dynamodb scan --table-name ai-video-codec-reasoning --limit 5
   ```
   Should show reasoning for each experiment.

3. **Check Experiment Logs:**
   New experiments should log:
   - "🤖 Running LLM pre-experiment analysis..."
   - "💡 LLM Hypothesis: Try X approach..."
   - "🤖 Running LLM post-experiment analysis..."

---

## Technical Details

### Compression Formula:
```python
reduction_percent = ((baseline - actual) / baseline) * 100
# If actual > baseline: negative (worse)
# If actual < baseline: positive (better)
```

### Current Experiment Flow:
```
1. Fetch past experiments from DynamoDB
2. LLM PRE-ANALYSIS → Generate hypothesis
3. Run procedural generation (NEEDS TO USE HYPOTHESIS)
4. Run neural network training
5. Upload results to S3 + DynamoDB
6. LLM POST-ANALYSIS → Analyze what happened
7. Store reasoning in DynamoDB
```

### Missing Link:
**Step 3 needs to read the hypothesis from step 2 and adjust its approach accordingly.**

---

## Summary

✅ **Fixed:** Compression display now clear (↑ = bad, ↓ = good)  
✅ **Fixed:** LLM analysis now runs before and after experiments  
✅ **Fixed:** Created script to analyze past experiments  
⚠️  **Still Need:** Agents must READ and APPLY LLM suggestions  
⚠️  **Still Need:** Results should vary between experiments  

The framework is now in place - agents just need to be connected to the LLM feedback loop!

