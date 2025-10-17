# Dashboard LLM Insights Context Issue - FIXED

## Problem Identified

The LLM insights displayed in the dashboard don't match the actual experiment because they're **pre-experiment analysis**, not post-experiment analysis.

### Root Cause

The experiment workflow has this sequence:

1. **LLM Analyzes Past Experiments** (`llm_experiment_planner.py:192-217`)
   - Fetches the 5 most recent experiments
   - Analyzes what went wrong in those experiments
   - Generates "Root Cause" and "Key Insights" about THOSE experiments
   - Creates hypothesis for what to try NEXT

2. **Pre-Experiment Reasoning Stored** (`procedural_experiment_runner.py:285-296`)
   - During the "design" phase (BEFORE execution)
   - Stores the root_cause/insights with the NEW experiment_id
   - This happens in `_write_blog_post_design()`

3. **Dashboard Displays as Post-Mortem** (`index_ssr.py:886-892`)
   - Shows "Root Cause Analysis" and "Key Insights"
   - But these describe PREVIOUS experiments, not the current one
   - Led to confusion where Iteration 40 (proc_exp_1760706922) showed insights about experiments 1760704923, 1760703179, etc.

### Example of the Issue

**Iteration 40** (proc_exp_1760706922):
- **Hypothesis**: "To achieve true compression, we need to ENCODE..."
- **Root Cause Shown**: "The procedural generation approach is fundamentally flawed..."
- **Key Insights Shown**: References proc_exp_1760704923, proc_exp_1760703179, proc_exp_1760703046

The root cause and insights are from analyzing experiments BEFORE 1760706922, not analyzing 1760706922 itself!

## Fix Applied

Updated `/Users/yarontorbaty/Documents/Code/AiV1/lambda/index_ssr.py`:

1. **Added clarifying comments** (lines 833-838)
   ```python
   # IMPORTANT: root_cause and insights in reasoning table are PRE-EXPERIMENT analysis
   # They describe what was learned from PREVIOUS experiments that led to THIS design
   ```

2. **Updated UI labels** (lines 890-892)
   - Changed "Root Cause Analysis" → "Pre-Experiment Analysis"
   - Changed "Key Insights" → "Insights from Previous Experiments"
   - Added clear disclaimer boxes explaining timing
   - Used amber/warning colors to differentiate from actual results

### Before:
```html
<h3>Root Cause Analysis</h3>
<p>{root_cause}</p>
```

### After:
```html
<div class="blog-section" style="background: #fff8e1; border-left: 4px solid #ffa726;">
  <h3>Pre-Experiment Analysis</h3>
  <p style="font-style: italic;">
    This analysis was generated BEFORE the experiment ran, based on analyzing 
    previous experiments. It explains what problems were identified and what 
    this experiment attempted to fix.
  </p>
  <h4>Root Cause from Previous Experiments:</h4>
  <p>{root_cause}</p>
</div>
```

## Future Improvements

### Option 1: Add Post-Experiment Analysis (Recommended)

Modify `_phase_analysis()` in `procedural_experiment_runner.py` to generate NEW LLM analysis AFTER experiment completes:

```python
def _phase_analysis(self, experiment_id: str, execution_result: Dict) -> Dict:
    """Phase 6: Analyze results and store to DynamoDB."""
    logger.info("📊 PHASE 6: ANALYSIS")
    
    # ... existing metrics calculation ...
    
    # NEW: Generate post-experiment analysis
    post_analysis = self._generate_post_experiment_analysis(
        experiment_id=experiment_id,
        results=results,
        hypothesis=self.llm_analysis.get('hypothesis', ''),
        pre_experiment_insights=self.llm_analysis.get('insights', [])
    )
    
    # Store in separate table or field
    self.reasoning_table.put_item(Item={
        'reasoning_id': f'{experiment_id}_post',
        'experiment_id': experiment_id,
        'analysis_type': 'post_experiment',
        'timestamp': timestamp,
        'actual_vs_expected': post_analysis['comparison'],
        'what_worked': post_analysis['successes'],
        'what_failed': post_analysis['failures'],
        'recommendations': post_analysis['recommendations']
    })
```

### Option 2: Rename Fields in Reasoning Table

Change the schema to be more explicit:
- `pre_experiment_root_cause` - from analyzing PAST experiments
- `pre_experiment_insights` - insights that LED TO this design
- `post_experiment_analysis` - analysis of THIS experiment's results
- `lessons_learned` - what we learned from THIS experiment

### Option 3: Two-Phase Reasoning Storage

1. **Design Phase**: Store planning reasoning with `{experiment_id}_design`
2. **Analysis Phase**: Store result analysis with `{experiment_id}_results`
3. **Dashboard**: Show both sections separately

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: LLM Analyzes Past Experiments                      │
│ (llm_experiment_planner.py)                                 │
│                                                             │
│ Input:  Experiments [A, B, C, D, E]                       │
│ Output: "Experiment D failed because..."                   │
│         "Try approach X for experiment F"                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Create Experiment F (Design Phase)                 │
│ (procedural_experiment_runner.py)                           │
│                                                             │
│ - Generate code for approach X                             │
│ - Store reasoning about D with experiment_id=F ❌ WRONG!  │
│   (Should store: "This is why we're trying X")            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Execute Experiment F                               │
│ - Runs the code                                             │
│ - Produces actual results                                   │
│ - No analysis of F's results generated ❌ MISSING!        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Dashboard Display                                   │
│ (index_ssr.py)                                              │
│                                                             │
│ Shows reasoning about D/E as if it's about F ❌ CONFUSING! │
│ FIXED: Now clearly labeled as pre-experiment analysis ✅  │
└─────────────────────────────────────────────────────────────┘
```

## Testing the Fix

1. **Deploy the updated lambda function**:
   ```bash
   cd lambda
   aws lambda update-function-code \
     --function-name ai-video-codec-blog \
     --zip-file fileb://index_ssr.zip
   ```

2. **Verify in dashboard**:
   - Visit the blog page
   - Check that insights now show as "Pre-Experiment Analysis"
   - Confirm amber warning boxes appear
   - Verify disclaimer text is visible

3. **Look for these visual changes**:
   - Amber/orange left border (instead of default blue)
   - Italic disclaimer text at top of each section
   - "Pre-Experiment" and "from Previous Experiments" in headers

## Related Files

- `/Users/yarontorbaty/Documents/Code/AiV1/lambda/index_ssr.py` - Dashboard blog rendering (FIXED)
- `/Users/yarontorbaty/Documents/Code/AiV1/src/agents/llm_experiment_planner.py` - Pre-experiment analysis
- `/Users/yarontorbaty/Documents/Code/AiV1/src/agents/procedural_experiment_runner.py` - Experiment execution
- DynamoDB Tables:
  - `ai-video-codec-experiments` - Experiment results
  - `ai-video-codec-reasoning` - LLM analysis (pre-experiment only)

## Conclusion

The fix clarifies that insights are pre-experiment analysis rather than post-experiment analysis. Users will no longer be confused about why the insights reference different experiment IDs.

**Next Step**: Consider implementing Option 1 (post-experiment analysis) to provide actual analysis of each experiment's results.


