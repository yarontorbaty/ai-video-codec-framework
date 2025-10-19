# 🔍 Failed Experiments Analysis

**Date:** October 18, 2025  
**Total Failed:** 42 experiments  
**Source:** Worker logs (actual execution errors)

---

## 📊 Top 3 Failure Reasons

### **1. Encoding Timeouts (30+ failures, ~71%)**

**Problem:** LLM-generated encoding code exceeds 120-second timeout

**Examples:**
```
Encoding timeout: Code execution exceeded 120 seconds
```

**Root Causes:**
- LLM generates code with inefficient loops
- Processing 300 HD frames (1920x1080) takes too long
- Complex algorithms without optimization
- Nested loops over large arrays

**Typical Pattern:**
```python
# LLM generates something like:
for frame in frames:  # 300 frames
    for y in range(1080):  # 1080 rows
        for x in range(1920):  # 1920 pixels
            for c in range(3):  # RGB channels
                # Complex operation
                # = 300 * 1080 * 1920 * 3 * operation
                # = 1.8 BILLION operations!
```

**Impact:** ~71% of failures

---

### **2. Dimension Mismatch Errors (6 failures, ~14%)**

**Problem:** Decoding produces frames with wrong dimensions

**Examples:**
```
ValueError: operands could not be broadcast together with shapes (540,960,3) (960,540,3)
ValueError: operands could not be broadcast together with shapes (1080,1920) (135,240)
ValueError: operands could not be broadcast together with shapes (1080,1920) (540,960)
ValueError: could not broadcast input array from shape (240,320,1,3) into shape (240,320,3)
```

**Root Causes:**
- Encoder downsamples video (e.g., 1920x1080 → 960x540)
- Decoder doesn't upscale back to original size
- Encoder/decoder dimension mismatch
- Incorrect reshape operations
- Wrong axis ordering (height vs width swapped)

**Typical Pattern:**
```python
# Encoding
compressed_frame = cv2.resize(frame, (960, 540))  # Downsample

# Decoding (WRONG - doesn't upscale back)
reconstructed = compressed_data  # Returns 960x540
# Should be: cv2.resize(compressed_data, (1920, 1080))
```

**Impact:** ~14% of failures

---

### **3. Missing Import/Name Errors (2+ failures, ~5%)**

**Problem:** LLM forgets to import required libraries or use correct names

**Examples:**
```
NameError: name 'np' is not defined
```

**Root Causes:**
- LLM generates code without proper imports
- Code references `np` but doesn't import `numpy as np`
- Inconsistent naming between encoder/decoder
- Copy-paste errors in LLM generation

**Typical Pattern:**
```python
# LLM generates:
def run_encoding_agent(frames, output_path):
    # Uses np.array() but never imported numpy!
    data = np.array(frames)  # NameError!
    
# Should be:
import numpy as np
def run_encoding_agent(frames, output_path):
    data = np.array(frames)  # Now works
```

**Impact:** ~5% of failures

---

## 📈 Failure Distribution

```
Encoding Timeouts:        30+ (71%) ████████████████████████████
Dimension Mismatches:      6  (14%) ██████
Missing Imports/Names:     2  ( 5%) ██
Other:                     4  (10%) ████
```

---

## 🎯 Recommended Fixes

### **Fix #1: Optimize Timeout Handling**

**Current:** 120-second timeout per operation  
**Problem:** Too generous, allows slow code to waste time

**Solution:**
1. Add complexity warnings to LLM prompt
2. Encourage vectorized operations over loops
3. Suggest downsampling before processing
4. Add example of efficient code

**Prompt Addition:**
```
⚠️ PERFORMANCE CRITICAL:
- Process 300 frames @ 1920x1080 (622 million pixels total)
- Avoid nested loops over pixels (use numpy vectorization)
- Example SLOW: for y in range(height): for x in range(width)
- Example FAST: frame * 0.5  (vectorized operation)
- Must complete in <60 seconds
```

---

### **Fix #2: Enforce Dimension Consistency**

**Current:** No validation of encoder/decoder output dimensions  
**Problem:** Decoder produces wrong-sized frames

**Solution:**
1. Add dimension check after decoding
2. Auto-resize if dimensions don't match
3. Add to LLM prompt: "Decoder must return frames of EXACT same size as input"

**Code Fix in `experiment_runner.py`:**
```python
def _execute_decoding(...):
    # After decoding
    decoded_frames = decode_func(...)
    
    # VALIDATE DIMENSIONS
    expected_shape = original_frames[0].shape
    if decoded_frames[0].shape != expected_shape:
        # Auto-fix: resize to correct dimensions
        decoded_frames = [cv2.resize(f, (expected_shape[1], expected_shape[0])) 
                         for f in decoded_frames]
```

**Prompt Addition:**
```
🎯 DIMENSION REQUIREMENTS:
- Input frames: 1920x1080 (width x height)
- Output frames: MUST be 1920x1080 (same as input)
- If you downsample during encoding, UPSAMPLE during decoding
- Use cv2.resize(frame, (1920, 1080)) to restore original size
```

---

### **Fix #3: Validate Imports**

**Current:** No import validation  
**Problem:** LLM generates code that references undefined names

**Solution:**
1. Add import validation before execution
2. Provide standard imports in prompt
3. Scan code for common names (np, cv2, pickle)

**Code Fix in `experiment_runner.py`:**
```python
def _validate_imports(code: str):
    """Check if code has required imports"""
    issues = []
    
    if 'np.' in code and 'import numpy' not in code:
        issues.append("Code uses 'np' but doesn't import numpy")
    if 'cv2.' in code and 'import cv2' not in code:
        issues.append("Code uses 'cv2' but doesn't import cv2")
    if 'pickle.' in code and 'import pickle' not in code:
        issues.append("Code uses 'pickle' but doesn't import pickle")
    
    return issues
```

**Prompt Addition:**
```
📦 REQUIRED IMPORTS:
Include these imports at the top of your code:
import cv2
import numpy as np
import pickle
```

---

## 🔄 Impact of Fixes

### **Expected Improvement:**

| Fix | Addresses | Est. Reduction |
|-----|-----------|----------------|
| Performance prompt | Timeouts (71%) | 50-70% reduction |
| Dimension validation | Mismatches (14%) | 90-100% reduction |
| Import checking | Name errors (5%) | 100% reduction |

**Overall:** Could reduce failures from ~42/50 (84%) to ~10/50 (20%)

---

## 📝 Summary Statistics

**Total Experiments:** ~50  
**Successful:** 8 (16%)  
**Failed:** 42 (84%)

**Failure Breakdown:**
1. **Encoding Timeouts:** 30+ (71% of failures)
   - Code too slow for 120-second limit
   - Inefficient nested loops
   - No vectorization

2. **Dimension Mismatches:** 6 (14% of failures)
   - Decoder returns wrong size frames
   - Missing upscaling after downsampling
   - Incorrect reshape operations

3. **Import Errors:** 2 (5% of failures)
   - Missing numpy import
   - Undefined variable names
   - LLM forgot required imports

4. **Other:** 4 (10% of failures)
   - Various edge cases
   - Syntax errors
   - File I/O issues

---

## 🎯 Action Items

### **High Priority (Immediate):**
1. ✅ Update LLM prompt with performance warnings
2. ✅ Add dimension validation to decoder
3. ✅ Add import validation before execution

### **Medium Priority (This week):**
1. Consider using smaller test video (30 frames instead of 300)
2. Add code complexity analyzer
3. Provide LLM with "good" and "bad" examples

### **Low Priority (Optional):**
1. Implement code caching to avoid re-running similar code
2. Add progressive timeout (30s, 60s, 120s based on complexity)
3. Create library of "known good" helper functions

---

**End of Analysis**

