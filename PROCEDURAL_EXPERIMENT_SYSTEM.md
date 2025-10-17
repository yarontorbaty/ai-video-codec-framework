# 🔄 Procedural Experiment System - No More Time Windows!

## ✅ System Redesign Complete

The experiment system has been completely redesigned from a **time-based** approach to a **procedural, step-by-step** approach with intelligent retry and self-healing.

---

## ❌ Old System (Time-Based):

- ⏱️  60-second windows between experiments
- ⏰ 10-minute timeout per experiment
- ❌ Failed experiments just logged and moved on
- ❌ No retry logic
- ❌ No self-healing
- ❌ Human never notified of issues

**Problems:**
- Experiments cut off mid-validation
- Framework bugs never fixed
- No feedback loop
- Wasted time on arbitrary delays

---

## ✅ New System (Procedural):

### 🎯 The 7-Step Process:

**1. Design Experiment and Code**
- LLM analyzes recent experiments
- Identifies issues and opportunities
- Generates improved codec code
- Uses tool calling to request framework fixes if needed

**2. Deploy to Sandbox**
- Code prepared for testing
- Verified and ready

**3. Validation (with retry)**
- Test code in sandbox (syntax, imports, safety)
- **If fails**: Analyze failure
  - ✅ Can auto-fix? → Use framework tools → Retry
  - ❌ Cannot fix? → Max retries (5) → Flag for human
- **If passes**: Move to execution

**4. Execution (with retry)**
- Run actual compression experiment
- Measure bitrate, quality, performance
- **If fails**: Analyze failure
  - ✅ Can auto-fix? → Use framework tools → Retry
  - ❌ Cannot fix? → Max retries (5) → Flag for human
- **If succeeds**: Move to analysis

**5. Analyze Results**
- Compare to previous experiments
- Determine if code should be adopted
- Store detailed metrics

**6. Design Next Experiment**
- Use results to inform next iteration
- Loop back to step 1

**7. Human Intervention (if needed)**
- Dashboard shows pulsing red "HUMAN NEEDED" badge
- Modal displays:
  - Which phase failed (validation/execution)
  - Why it failed
  - What was attempted
  - Specific failure details
- Human can review and fix manually

---

## 🛠️ Key Features:

### 1. **No Time Limits**
- Experiments run until completion
- Validation and execution retry until success or max attempts
- System works through problems methodically

### 2. **Intelligent Retry**
- Max 5 validation retries
- Max 5 execution retries
- Each retry includes failure analysis
- Framework modifications applied between retries

### 3. **Self-Healing with Tools**
- Uses `FrameworkModifier` tools
- Can fix sandbox restrictions
- Can install missing packages
- Can modify framework code
- Restarts services after fixes

### 4. **Human Intervention Tracking**
- `needs_human` flag in experiment data
- `human_intervention_reasons` array with details
- Dashboard alert system
- Pulsing red badge for visibility

### 5. **Brief Pauses Only**
- 10s pause between successful experiments (monitoring)
- 30s pause between failed experiments (brief cooldown)
- 5min pause after 5 consecutive failures (recovery mode)

---

## 📊 What Changed:

### Files Created:
- `src/agents/procedural_experiment_runner.py` - New runner with 7-step process

### Files Modified:
- `scripts/autonomous_orchestrator_llm.sh` - Uses new procedural runner, removed time windows
- `lambda/admin_api.py` - Tracks `needs_human` and `human_intervention_reasons`
- `dashboard/admin.js` - Shows human intervention alerts with modal

---

## 🚀 Deployment Status:

**Ready to deploy!** Files are prepared and need to be pushed to orchestrator and dashboard.

---

## 📈 Expected Improvements:

### Before (Time-Based):
- ❌ Experiments cut off mid-process
- ❌ No retry on failures
- ❌ Framework bugs accumulate
- ❌ Human unaware of issues
- ❌ Wasted time on delays

### After (Procedural):
- ✅ Experiments run to completion
- ✅ Automatic retry with fixes
- ✅ Self-healing framework
- ✅ Clear human intervention alerts
- ✅ Minimal wasted time

---

## 🔍 Monitoring:

### Dashboard Indicators:
- **Green**: Experiments completing successfully
- **Red Pulsing Badge**: Human intervention needed
- **Analysis Column**: Failure details for each experiment
- **Human Column**: Shows which experiments need attention

### Log Indicators:
```
📐 PHASE 1: DESIGN
📦 PHASE 2: DEPLOY
🔍 PHASE 3: VALIDATION (with intelligent retry)
▶️  PHASE 4: EXECUTION (with intelligent retry)
📊 PHASE 5: ANALYSIS
✅ EXPERIMENT COMPLETED SUCCESSFULLY

Or:

🚨 HUMAN INTERVENTION REQUIRED:
  - validation: Max validation retries exceeded
  - execution: Max execution retries exceeded
```

---

## 🎯 Next Steps:

1. **Deploy to orchestrator**:
   ```bash
   ./scripts/deploy_meta_autonomy.sh
   ```
   
2. **Upload dashboard files**:
   ```bash
   ./scripts/deploy_admin.sh
   ```

3. **Monitor first procedural experiment**:
   - Check logs: `tail -f /tmp/orch.log`
   - Watch dashboard: https://aiv1codec.com/admin
   - Look for "PHASE X" messages

4. **Verify self-healing**:
   - Introduce a sandbox restriction
   - Watch LLM detect and fix it via tools
   - Confirm retry succeeds

---

## 💡 Examples:

### Successful Experiment Flow:
```
ITERATION 3 - STARTING
📐 PHASE 1: DESIGN
  ✅ Design complete
📦 PHASE 2: DEPLOY
  ✅ Code ready for deployment
🔍 PHASE 3: VALIDATION (with intelligent retry)
  Validation attempt 1/5
  ✅ Validation PASSED on attempt 1
▶️  PHASE 4: EXECUTION (with intelligent retry)
  Execution attempt 1/5
  ✅ Execution SUCCEEDED on attempt 1
  Bitrate: 4.2341 Mbps
📊 PHASE 5: ANALYSIS
  ✅ Results stored to DynamoDB
  ✅ Analysis complete
✅ EXPERIMENT COMPLETED SUCCESSFULLY
ITERATION 3 - COMPLETED
```

### Failed with Self-Healing:
```
ITERATION 4 - STARTING
📐 PHASE 1: DESIGN
  ✅ Design complete
📦 PHASE 2: DEPLOY
  ✅ Code ready for deployment
🔍 PHASE 3: VALIDATION (with intelligent retry)
  Validation attempt 1/5
  ❌ Validation FAILED on attempt 1
  Failure: import_error
  Root cause: Module 'scipy' not found
  Fix: Install scipy package
  🔧 Attempting auto-fix...
  🛠️  LLM requested tool: install_python_package
  ✅ Auto-fix applied, retrying...
  Validation attempt 2/5
  ✅ Validation PASSED on attempt 2
▶️  PHASE 4: EXECUTION (with intelligent retry)
  Execution attempt 1/5
  ✅ Execution SUCCEEDED on attempt 1
✅ EXPERIMENT COMPLETED SUCCESSFULLY
```

### Needs Human:
```
ITERATION 5 - STARTING
🔍 PHASE 3: VALIDATION (with intelligent retry)
  Validation attempt 1/5
  ❌ Validation FAILED
  Validation attempt 2/5
  ❌ Validation FAILED
  ...
  Validation attempt 5/5
  ❌ Validation FAILED
  ❌ Validation failed after 5 attempts
🚨 HUMAN INTERVENTION REQUIRED:
  - validation: Max validation retries exceeded
❌ EXPERIMENT FAILED: validation_failed
```

---

**The system is now truly procedural - no artificial time limits, just thorough, intelligent problem-solving!** 🎯

