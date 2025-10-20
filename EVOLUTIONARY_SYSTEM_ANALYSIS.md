# Evolutionary System Analysis & Fix

## 📊 Problem: System Regression Instead of Improvement

### Initial Observation
- **Early Gens (0-3):** Avg score 254
- **Late Gens (45-49):** Avg score 167  
- **Change:** -34% regression ❌

Instead of smooth improvement, saw chaotic oscillation between good and terrible results.

---

## 🔍 Root Cause Analysis

### The Bug (Line 238 in `evolutionary_orchestrator.py`)

**WRONG CODE:**
```python
# Only sorted by compression ratio
experiments.sort(key=lambda x: x.get('compression_ratio', 0), reverse=True)
```

**What This Selected:**
- Codec with 735x compression, 6663 MSE, 9.89 PSNR → #1
- Codec with 532x compression, 5135 MSE, 10.91 PSNR → #2

**Result:** Claude was told "extreme compression is best" regardless of quality!

---

## 📈 What Actually Happened

### Generation Pattern
```
Gen 1:  1278 score ✅ → Gen 2:  78 score ❌ (crash)
Gen 3:  483 score  ✅ → Gen 4:  23 score ❌ (crash) 
Gen 7:  1265 score ✅ → Gen 8:  598 score ❌ (worse)
Gen 9:  1545 score ✅ → Gen 10: 100 score ❌ (crash)
```

### Why?
1. **Gen 0:** Random exploration found some extreme compression (low quality)
2. **Gen 1+:** System told Claude: "These high-compression codecs are best!"
3. **Claude:** Optimized for compression only, ignoring quality
4. **Result:** Chaotic oscillation - when it found balanced solutions, it was by accident

---

## ✅ The Fix

### Changed Selection Criteria

**NEW CODE:**
```python
# Calculate performance score for each experiment
for exp in experiments:
    psnr = float(exp.get('metrics', {}).get('psnr_db', 0))
    compression = float(exp.get('metrics', {}).get('compression_ratio', 1))
    exp['_performance_score'] = psnr * compression

# Sort by performance score (balances quality and compression)
experiments.sort(key=lambda x: x.get('_performance_score', 0), reverse=True)
```

### Updated Claude Prompt

**Now tells Claude:**
```
GOAL: Maximize Performance Score = PSNR × Compression Ratio
- High PSNR means good quality (low distortion)
- High compression means small file size  
- You need BOTH to get a high score!

STRATEGIES:
- Balance is key - 100x compression with 20 PSNR beats 500x with 5 PSNR!
- Study the top performers' techniques and try variations
- If high compression but low PSNR, try to improve quality
- If high PSNR but low compression, try to compress more
```

---

## 📊 Metrics Being Tracked

### Source Video
- **Size:** 122,880 bytes (120 KB) uncompressed
- **Bitrate:** 2.95 Mbps @ 30 fps
- **Dimensions:** 64×64 pixels, 10 frames

### All Experiments Track:
1. **PSNR** (dB) - quality metric
2. **SSIM** - perceptual quality
3. **MSE** - pixel error  
4. **Compression Ratio** - file size reduction
5. **Bitrate (Mbps)** - bandwidth required

### Dashboard Ranking Formula:
```
Performance Score = PSNR × Compression Ratio
```

This rewards:
- ✅ High compression WITH good quality
- ❌ Penalizes extreme compression with terrible quality  
- ✅ Guides toward balanced solutions

### Top Performers (After Migration):
- **Best:** Gen 9, exp 0 → 0.004 Mbps (99.86% bandwidth reduction)
- **Top 10 Avg:** 0.033 Mbps (~1% of source, 99% reduction)
- **Score Range:** 600-1545 (vs old system: ~150 max)

---

## 🎯 Expected Outcome After Fix

### Smooth Progression (not chaotic oscillation):
- **Gen 0:** ~200 avg score (random baseline)
- **Gen 10:** ~400 avg score (learning patterns)
- **Gen 25:** ~700 avg score (refined techniques)
- **Gen 50:** ~1200 avg score (optimized codecs)

### Key Improvements:
✅ **Proper evolutionary feedback** - Claude learns from truly best solutions
✅ **Balanced optimization** - Quality + compression, not just compression
✅ **Aligned metrics** - Selection matches dashboard ranking  
✅ **Clear guidance** - Claude knows the goal is balance

---

## 📁 Files Modified

1. **`v3/orchestrator/evolutionary_orchestrator.py`**
   - Fixed `_get_top_performers()` to use performance score
   - Updated logging to show score breakdown
   - Improved prompt to explain PSNR × compression goal

2. **`v3/worker/fast_main.py`**
   - Fixed `_store_results_batch()` to store metrics in proper nested schema
   - Calculates PSNR, SSIM, bitrate from MSE and compression

3. **`migrate_fast_experiments.py`**
   - Migrated 13,689 experiments to new schema
   - Converted flat `mse`/`compression_ratio` → nested `metrics` object

---

## 🚀 Next Steps

1. **Deploy fixed orchestrator** to EC2
2. **Start new evolutionary run** (50 generations)
3. **Monitor for smooth improvement** (not oscillation)
4. **Compare Gen 0 vs Gen 50** to validate evolutionary learning

Expected runtime: ~2.5 hours for 50 generations @ 100 experiments/gen

---

## 📌 Key Takeaway

**The system wasn't broken - the feedback was!**

By optimizing for compression only, we accidentally told Claude to ignore quality. Now that selection criteria matches the dashboard ranking (PSNR × compression), the evolutionary feedback loop should work as intended.

