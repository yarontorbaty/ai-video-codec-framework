# PVC v2.0 Phase 1 Evaluation Results

**Date:** October 19, 2025  
**Model:** Extended (47 functions)  
**Status:** ⚠️ Function prediction working, but parameter prediction needed

---

## 📊 Step A Results: Evaluation

### Model Performance

**Training Set:**
- Final accuracy: **53.2%**
- Loss: **0.5280**
- Training samples: 1,000
- Convergence: Good (25.9% → 53.2%)

**Test Set:**
- Function prediction accuracy: **6.8% ± 11.7%**  
- **⚠️ Major overfitting detected!**

### Analysis

The **huge gap** between train (53%) and test (6.8%) accuracy indicates:

1. **Overfitting:** Model memorized training samples rather than learning general patterns
2. **Insufficient training data:** 1,000 samples is too few for 47 functions
   - Need ~100-200 samples per function = 5,000-10,000 total
3. **Distribution mismatch:** Test data generation might differ from training

### Visual Quality

**PSNR/SSIM measurement inconclusive:**
- Used average-color reconstruction (not actual function execution)
- PSNR: `inf` (some frames identical by chance)
- SSIM: 0.84 (meaningless without proper reconstruction)

**Conclusion:** Cannot measure visual quality without implementing:
1. Full reconstruction pipeline (execute predicted functions)
2. Parameter prediction (current model only predicts function IDs)

---

## 🎯 Step B: Parameter Supervision (REQUIRED)

### Why Parameter Supervision is Critical

**Current state:**
- ✅ Model predicts: "Use `draw_circle` function"
- ❌ Model doesn't predict: WHERE to draw it, what COLOR, what SIZE

**With parameter supervision:**
- ✅ Model predicts: "`draw_circle` at (128, 128), radius 50, color (255, 0, 0)"
- ✅ Can actually reconstruct the image
- ✅ Can measure meaningful PSNR/SSIM

### Implementation Plan

**Architecture Changes:**
1. **Sequence-level parameter prediction**
   - Current: Only predicts params for first function
   - Needed: Predict params for ALL functions in sequence
   
2. **Per-position parameter head**
   - For each sequence position, predict 10 parameters
   - Output shape: `(batch, seq_len, 10)`
   
3. **Combined loss function**
   - Function loss: CrossEntropy for function IDs
   - Parameter loss: MSE for (coords, colors)
   - Total: `0.5 × func_loss + 0.5 × param_loss`

**Training Changes:**
1. **More training data:** 5,000-10,000 samples (vs current 1,000)
2. **Longer sequences:** 20-30 functions per frame (vs current 3-9)
3. **Better regularization:** Dropout, weight decay to prevent overfitting

**Expected Results:**
- Function accuracy: 40-50% (slightly lower due to harder task)
- Parameter MSE: < 0.05 (normalized coords/colors)
- **PSNR: 10-20 dB** (with full reconstruction)
- **SSIM: 0.7-0.85** (good structural similarity)

### Estimated Effort

- **Code changes:** 2-3 hours
  - Modify `ParameterPredictor` to predict full sequence
  - Update training loop for combined loss
  - Implement reconstruction pipeline
  
- **Training time:** 30-60 minutes (5K samples, 30 epochs)
  
- **Total:** 3-4 hours to complete

---

## 💡 Recommendations

### Immediate Actions:

1. **✅ Implement parameter supervision** (CRITICAL)
   - Without this, we cannot achieve target PSNR
   - Function IDs alone are insufficient
   
2. **✅ Increase training data to 5K samples**
   - Reduces overfitting
   - Better generalization
   
3. **✅ Implement reconstruction pipeline**
   - Execute predicted functions with predicted parameters
   - Enables meaningful PSNR measurement

### Alternative Approaches (if parameter supervision doesn't work):

**Plan B: Reduce function count**
- Keep only 20-30 most useful functions
- Easier to learn with limited data
- Still 2-3x more expressive than baseline

**Plan C: Hybrid approach**
- Use procedural for coarse structure (5-10 functions)
- Add neural residual encoder for details
- Target: 92-96% compression with PSNR 30-40 dB

---

## 🚀 Next Steps

### Option 1: Full Implementation (Recommended)
**Time:** 4 hours  
**Expected PSNR:** 10-20 dB

1. Implement sequence-level parameter prediction (2 hrs)
2. Train with 5K samples (1 hr)
3. Implement reconstruction + eval (1 hr)
4. Measure results

### Option 2: Quick Validation
**Time:** 1 hour  
**Expected:** Proof that approach can work

1. Train with 5K samples (function ID only) (30 min)
2. Check if test accuracy improves to 30-40% (30 min)
3. If yes → proceed with full implementation
4. If no → consider Plan B/C

### Option 3: Pivot to Hybrid
**Time:** 6 hours  
**Expected PSNR:** 30-40 dB (more reliable)

1. Use PVC for coarse structure only (5-10 functions)
2. Add neural residual encoder for fine details
3. Two-stage codec: procedural + residual
4. More likely to hit quality targets

---

## 📋 Summary

**Phase 1 Status:**
- ✅ Function library expanded (47 functions)
- ✅ Model architecture working
- ⚠️ Overfitting detected (53% train, 6.8% test)
- ❌ **Cannot measure PSNR without parameter prediction**

**Critical Path:**
Parameter supervision is **mandatory** to achieve any meaningful visual quality. Without it, we have function IDs but no way to execute them (don't know where/how to draw).

**Recommendation:**
Proceed with **Option 1** (full implementation). The infrastructure is ready, we just need to:
1. Fix parameter prediction
2. Add more training data
3. Implement reconstruction

**Expected outcome:** PSNR 10-20 dB within 4 hours of work.

