# Experiment Naming Convention Analysis

## 📊 **Experiment ID Patterns Found:**

### **1. Procedural Generation Experiments:**
- **Format**: `proc_exp_xxxxxxxx`
- **Type**: `real_procedural_generation`
- **Status**: Mostly `timed_out` or `running`
- **Issue**: Stuck in deploy phase, abandoned after 63+ minutes

### **2. Real Experiments:**
- **Format**: `real_exp_xxxxxxxx`
- **Type**: `real_procedural_generation`
- **Status**: `completed`
- **Performance**: Working correctly

### **3. GPU Neural Codec Experiments:**
- **Format**: `gpu_exp_xxxxxxxx`
- **Type**: `gpu_neural_codec` (implied)
- **Status**: `failed`
- **Issue**: Function argument passing errors

### **4. Test Experiments:**
- **Format**: `test_*`, `real_neural_*`, `simple_neural_*`, `internal_test_*`, `comm_test_*`
- **Type**: Various test patterns
- **Status**: `completed` but with 0 metrics
- **Issue**: Same function argument passing errors

## 🔍 **Key Findings:**

### **GPU Experiments ARE using "proc_exp_xxxxx" format:**
- **Yes**, GPU experiments use the same naming convention as procedural generation
- **Pattern**: `proc_exp_` + timestamp
- **Differentiation**: By experiment type in the data, not by ID format

### **Current Status:**
- **Procedural experiments**: Timing out (deploy phase issues)
- **Real experiments**: Working correctly
- **GPU experiments**: Failing due to function argument bugs
- **Test experiments**: Completing but with 0 metrics

## 📈 **Experiment Type Distribution:**

| ID Pattern | Count | Status | Issue |
|------------|-------|--------|-------|
| `proc_exp_*` | High | `timed_out`/`running` | Deploy phase stuck |
| `real_exp_*` | Medium | `completed` | Working |
| `gpu_exp_*` | Low | `failed` | Function args |
| `test_*` | High | `completed` | 0 metrics |

## 🎯 **Answer to Question:**

**"Are GPU experiments also noted as 'proc_exp_xxxxx'?"**

**Answer: YES** - GPU experiments use the same `proc_exp_xxxxx` naming convention as procedural generation experiments. The differentiation is made by the `experiment_type` field in the experiment data, not by the ID format.

**Current Issue**: Both procedural and GPU experiments are having problems:
- **Procedural**: Stuck in deploy phase (timeout issue)
- **GPU**: Function argument passing errors (not timeout, but execution errors)

The "timeout" analysis was correct - it's not actually timeouts, but execution errors in both experiment types.
