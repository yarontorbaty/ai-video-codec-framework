# GPU EXPERIMENT 503 ERROR - ROOT CAUSE IDENTIFIED

## 🎯 **ROOT CAUSE: Worker Busy Processing Previous Experiment**

### **The Issue:**
The worker is configured with a single-threaded processing model:
- Line 371-375 in `workers/neural_codec_http_worker.py`: Worker checks `is_processing` flag
- If `True`, returns 503 Service Unavailable
- The CPU experiment sets `is_processing = True` and processes in background thread
- The GPU experiment arrives while CPU is still processing → 503 error

### **Code Analysis:**

```python
@app.route('/experiment', methods=['POST'])
def handle_experiment():
    """Handle experiment job request."""
    try:
        if worker.is_processing:  # LINE 371 - BLOCKS SECOND REQUEST
            return jsonify({
                'status': 'busy',
                'message': 'Worker is currently processing another experiment'
            }), 503  # RETURNS 503 FOR GPU EXPERIMENT
```

### **Why It Happens:**
1. CPU experiment submitted → `is_processing = True`
2. Worker processes CPU experiment in background thread
3. CPU experiment takes > 10 seconds to complete
4. GPU experiment arrives while `is_processing = True`
5. Worker returns 503 immediately without queuing

### **Why 10 Seconds Isn't Enough:**
- CPU experiment processing time is longer than expected
- The worker needs to:
  - Execute encoding agent code
  - Execute decoding agent code
  - Calculate metrics
  - Send result to orchestrator
  - All of this takes > 10 seconds

### **Why Experiments Aren't Completing:**
The experiments ARE completing (worker processes them), but they're not being saved to DynamoDB because:
1. The worker sends results to orchestrator at `http://10.0.1.109:8081/experiment_result`
2. The orchestrator may not have this endpoint implemented
3. OR the orchestrator isn't saving results to DynamoDB

## 🔧 **SOLUTIONS:**

### **Option A: Queue-Based Processing (RECOMMENDED)**
Replace the 503 rejection with a queue:
```python
if worker.is_processing:
    # Queue the experiment instead of rejecting
    worker.experiment_queue.append(job_data)
    return jsonify({
        'status': 'queued',
        'queue_position': len(worker.experiment_queue)
    }), 202
```

### **Option B: Increase Delay Between Experiments**
Wait longer (30+ seconds) between CPU and GPU experiments:
```python
time.sleep(30)  # Wait for CPU to complete
```

### **Option C: Parallel Processing**
Allow multiple experiments to run simultaneously (requires thread-safe code)

### **Option D: Separate CPU and GPU Workers**
Run two worker instances - one for CPU, one for GPU

## 📊 **DIAGNOSIS COMPLETE**

**Status**: ROOT CAUSE IDENTIFIED ✅

**Issue**: Worker rejects second experiment with 503 because first experiment is still processing

**Fix Required**: Implement experiment queue OR ensure experiments run sequentially with sufficient delay

**Current Behavior**: 
- CPU experiment: ACCEPTED → Processing (takes > 10s)
- GPU experiment: REJECTED (503) → Worker busy

**Desired Behavior**:
- CPU experiment: ACCEPTED → Processing → Complete
- GPU experiment: ACCEPTED → Queued/Processing → Complete

## 🎯 **RECOMMENDATION:**

Implement a simple experiment queue in the worker to handle multiple requests gracefully instead of rejecting with 503.
