# Autonomous Monitoring & Improvement Plan

## Current Status

**As of:** 2025-10-17 06:50 UTC

### ✅ What's Fixed
- System NOW uses LLM-generated compression code
- Execution phase properly calls `run_real_experiment_with_code()`
- All previous experiments (26) purged for fresh start
- Orchestrator running with fix deployed

### 🎯 Expected Behavior
- Each experiment should produce DIFFERENT bitrates
- Not all 15.04 Mbps anymore
- Some experiments should beat 10 Mbps baseline
- Progressive improvement toward 1 Mbps goal

---

## Hourly Monitoring Checklist

### 1. Check Progress (Every Hour)

```python
import boto3
from decimal import Decimal

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('ai-video-codec-experiments')

response = table.scan()
items = [Decimal_to_float(item) for item in response.get('Items', [])]
completed = [i for i in items if i.get('status') == 'completed']

if completed:
    bitrates = []
    for item in completed:
        try:
            import json
            exp_data = json.loads(item['experiments'])
            metrics = exp_data[0]['real_metrics']
            bitrates.append(metrics['bitrate_mbps'])
        except: pass
    
    if bitrates:
        print(f"Experiments: {len(bitrates)}")
        print(f"Bitrates: {bitrates}")
        print(f"Average: {sum(bitrates)/len(bitrates):.2f} Mbps")
        print(f"Best: {min(bitrates):.2f} Mbps")
        print(f"Variety: {len(set(bitrates))} unique values")
```

### 2. Key Metrics to Track

**✅ SUCCESS INDICATORS:**
- Bitrates are DIFFERENT (not all the same)
- Average trending DOWN toward 1 Mbps
- Best result < 10 Mbps (beating baseline)
- 3+ unique bitrate values

**🔴 FAILURE INDICATORS:**
- All experiments same bitrate → LLM code not being used
- Average trending UP → LLM getting worse
- Many timeouts → validation/execution issues
- Same errors repeated → systemic bug

### 3. When to Intervene

**PURGE & RESTART if:**
- 10+ experiments, all identical bitrates
- No experiments completing (all timeout)
- Best result > 15 Mbps after 5+ experiments

**FIX CODE if:**
- Repeated validation errors in logs
- Execution failures with same error
- Human intervention needed repeatedly

**LET IT RUN if:**
- Bitrates vary
- Some < 10 Mbps
- Trend improving

---

## Automated Monitoring (Optional)

### Option 1: CloudWatch Alarm

```yaml
Alarm:
  MetricName: CompletedExperiments
  Threshold: 0
  Period: 3600  # 1 hour
  Action: SNS notification
```

### Option 2: Cron Job

```bash
# /etc/cron.d/ai-codec-monitor
0 * * * * python3 /path/to/monitor_script.py
```

### Option 3: Lambda Scheduled Function

```python
# Runs every hour, checks progress, sends report
def handler(event, context):
    # Analyze experiments
    # Send to SNS/email
    # Trigger fixes if needed
```

---

## Specific Issues to Watch For

### Issue 1: Identical Bitrates (CRITICAL)
**Symptom:** All experiments produce same Mbps  
**Cause:** LLM code not being executed  
**Fix:** Check `run_real_experiment_with_code()` is being called  
**Action:** Review execution logs, redeploy if needed

### Issue 2: No Progress After 10 Experiments
**Symptom:** Best result still > 10 Mbps  
**Cause:** LLM not learning or bad feedback loop  
**Fix:** Check LLM is seeing previous results, adjust prompt  
**Action:** Review `llm_experiment_planner.py` analysis logic

### Issue 3: High Timeout Rate (>20%)
**Symptom:** Many experiments stuck in validation/execution  
**Cause:** LLM generating invalid/slow code  
**Fix:** Tighten sandbox restrictions, add timeouts  
**Action:** Review timeout thresholds, add auto-fix logic

### Issue 4: Validation Failures
**Symptom:** Many experiments need human intervention  
**Cause:** LLM code not meeting sandbox requirements  
**Fix:** Update LLM prompt with sandbox rules  
**Action:** Add sandbox documentation to system prompt

---

## Expected Timeline

### Hour 1 (First 3-5 experiments)
- **Expected:** Varying bitrates, some > 15, some < 15
- **Best case:** One experiment < 10 Mbps
- **Worst case:** All still ~15 Mbps (fix failed)

### Hour 2 (6-10 experiments)
- **Expected:** Trend emerging, best result improving
- **Best case:** Best < 5 Mbps, clear improvement pattern
- **Worst case:** No clear trend, random variation

### Hour 3 (11-15 experiments)
- **Expected:** Best result < 5 Mbps, LLM learning
- **Best case:** Approaching 1 Mbps, clear strategy
- **Worst case:** Plateaued at 10+ Mbps

### Hour 4-6 (16-30 experiments)
- **Expected:** Progressive improvement, exploring variations
- **Best case:** Hit < 1 Mbps target!
- **Worst case:** Stuck at local minimum (5-10 Mbps)

### Hour 7-8 (31+ experiments)
- **Expected:** Refinement, consistency
- **Best case:** Reliably < 1 Mbps, optimizing further
- **Worst case:** Need to reset approach

---

## Emergency Procedures

### If System Crashes
```bash
# SSH to orchestrator
ssh -i /path/to/key ec2-user@<instance-ip>

# Check logs
tail -100 /tmp/orch.log

# Restart
pkill -f autonomous_orchestrator_llm.sh
cd /home/ec2-user/ai-video-codec
nohup bash scripts/autonomous_orchestrator_llm.sh > /tmp/orch.log 2>&1 &
```

### If Needs Code Fix
```bash
# Local
cd /Users/yarontorbaty/Documents/Code/AiV1
# Edit code
git add -A && git commit -m "Fix: description" && git push

# Deploy
tar -czf /tmp/fix.tar.gz src/
aws s3 cp /tmp/fix.tar.gz s3://ai-video-codec-artifacts-580473065386/fix.tar.gz

# Update orchestrator (via SSM or SSH)
```

### If Needs Fresh Start
```python
# Purge experiments
python3 -c "import boto3; ..."  # See purge script above

# Verify purge
# Check DynamoDB is empty

# Restart orchestrator
# Will start at iteration 1
```

---

## Success Criteria

**After 8 hours, system should have:**
- ✅ 20-40 completed experiments
- ✅ Varying bitrates (not all same)
- ✅ Best result < 5 Mbps (50% better than baseline)
- ✅ Clear improvement trend
- ✅ < 20% timeout rate
- ✅ Evidence LLM is learning from results

**Stretch goal:**
- 🎯 Best result < 1 Mbps (TARGET ACHIEVED!)

---

## Notes for Morning Review

When you wake up, check:

1. **Dashboard** → How many experiments?
2. **Blog** → Are results showing?
3. **Best bitrate** → What's the lowest achieved?
4. **Trend** → Improving, plateaued, or random?
5. **Logs** → Any errors or issues?
6. **Database** → Clean data?

Run analysis script:
```bash
cd /Users/yarontorbaty/Documents/Code/AiV1
# Will create analysis when you ask for it
```

---

**Status:** Fresh start with fix deployed  
**Goal:** Autonomous improvement toward 1 Mbps  
**Timeline:** 8 hours overnight  
**Success metric:** Best < 5 Mbps, trend improving

