# Adaptive Evolution System

## Overview
The AI Video Codec framework now includes **automatic architecture evolution** - the system can rewrite its own compression algorithms based on LLM suggestions.

## How It Works

### 1. **LLM Analysis**
- After each experiment, Claude Sonnet 4.5 analyzes the results
- Identifies root causes of failures
- Generates specific code suggestions for better compression

### 2. **Code Generation**
- LLM generates new Python compression functions
- Code follows specific interface: `compress_video_frame(frame, frame_index, config)`
- Includes validation and safety checks

### 3. **Automatic Evaluation**
- `AdaptiveCodecAgent` tests the new code in a secure sandbox
- Runs on multiple test frames (random, black, gray)
- Calculates compression metrics (bitrate, compression ratio)

### 4. **Adoption Decision**
The system automatically adopts new code if it's better:
- **10% better bitrate** → Adopt
- **20% better compression ratio** → Adopt
- Otherwise → Keep current implementation

### 5. **Continuous Evolution**
- Each adopted implementation becomes the new baseline
- Version number increments
- Performance history is tracked
- System evolves towards better compression

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Experiment Orchestrator                 │
└────────────┬───────────────────────────────────┬────────┘
             │                                   │
             v                                   v
    ┌────────────────┐                  ┌──────────────┐
    │  LLM Planner   │                  │  Adaptive    │
    │                │                  │  Codec Agent │
    │ - Analyze      │                  │              │
    │ - Generate Code│─────Code─────────>│ - Test       │
    │ - Suggest      │                  │ - Evaluate   │
    └────────────────┘                  │ - Adopt/Reject│
                                        └──────┬───────┘
                                               │
                                               v
                                        ┌──────────────┐
                                        │   Secure     │
                                        │   Sandbox    │
                                        │ (CodeSandbox)│
                                        └──────────────┘
```

## Key Components

### `AdaptiveCodecAgent`
**Location:** `src/agents/adaptive_codec_agent.py`

**Methods:**
- `evolve_with_llm_code(generated_code_info)` - Main evolution entry point
- `test_generated_code(code, function_name)` - Validate and test code
- `should_adopt_new_code(new_metrics)` - Decision logic
- `save_implementation(code, metrics)` - Persist new codec version
- `compress_video(input_path, output_path)` - Use current best implementation

**Persistence:**
- Current best implementation saved to `/tmp/best_codec_implementation.json`
- Survives between experiments
- Version-tracked for rollback if needed

### `CodeSandbox`
**Location:** `src/utils/code_sandbox.py`

**Features:**
- AST-based validation (no forbidden imports/functions)
- Resource limits (timeout, memory)
- Separate process execution for isolation
- Allowed modules: `numpy`, `cv2`, `math`, `json`, `struct`, `base64`, `typing`

## Evolution Criteria

### Performance Metrics
1. **Compression Ratio** - Original size / Compressed size
2. **Bitrate (Mbps)** - Bits per second for 30fps video
3. **Success Rate** - % of test frames successfully compressed

### Adoption Thresholds
```python
# Adopt if:
new_bitrate < prev_bitrate * 0.9  # 10% bitrate improvement
# OR
new_compression_ratio > prev_compression_ratio * 1.2  # 20% compression improvement
```

## Example Evolution Cycle

### Iteration 1: Baseline
```
Status: No implementation
Result: Uses fallback (copy input)
Bitrate: 15 Mbps (baseline HEVC × 1.5)
```

### Iteration 2: LLM Suggests DCT Compression
```
LLM Analysis: "System generates new content, doesn't compress"
Generated Code: Simple DCT-based compression
Test Results: 8.5 Mbps
Decision: ✅ ADOPTED (43% better than baseline)
Version: 1
```

### Iteration 3: LLM Suggests Quantization
```
LLM Analysis: "High-frequency information is redundant"
Generated Code: DCT + Aggressive quantization
Test Results: 2.1 Mbps
Decision: ✅ ADOPTED (75% better than v1)
Version: 2
```

### Iteration 4: LLM Suggests Neural Encoding
```
LLM Analysis: "Need learned features, not hand-crafted DCT"
Generated Code: Pretrained ResNet features + quantization
Test Results: 1.8 Mbps
Decision: ✅ ADOPTED (14% better than v2)
Version: 3
```

## Monitoring Evolution

### Dashboard Updates
The dashboard will show:
- **Evolution Events**: When new code is adopted
- **Version Number**: Current codec implementation version
- **Adoption Rate**: % of LLM suggestions that are adopted
- **Performance Trajectory**: Bitrate improvement over time

### Log Messages
```
🧬 Evaluating LLM-generated code for evolution...
📊 Metrics: 4.23x compression, 2.14 Mbps
🎯 New code is 45.2% better!
💾 Saved new implementation v4
🎉 EVOLUTION SUCCESS! Adopted new implementation v4
```

## Safety & Rollback

### Validation
- All code validated with AST before execution
- Forbidden operations blocked (file I/O, network, subprocess)
- Timeout protection (30 seconds per test)

### Rollback
If new code causes failures:
1. System keeps previous version in memory
2. Can manually restore from version history
3. Performance history shows degradation

### Version Management
```json
{
  "version": 4,
  "code": "def compress_video_frame(...)...",
  "metrics": {
    "compression_ratio": 4.23,
    "bitrate_mbps": 2.14
  },
  "timestamp": "2025-10-16T21:15:30"
}
```

## Expected Outcomes

### Short Term (Days 1-2)
- LLM identifies the core architecture flaw
- Generates simpler, correct compression approaches
- System adopts 2-3 better implementations
- Bitrate drops from 15 Mbps → 5-8 Mbps

### Medium Term (Days 3-5)
- LLM suggests learned features (ResNet, etc.)
- System integrates neural components correctly
- Bitrate reaches 1-3 Mbps range
- PSNR validation added

### Long Term (Days 6-7)
- LLM proposes hybrid approaches (neural + procedural)
- System approaches target: <1 Mbps, PSNR >95%
- Codec evolves to competitive performance

## Comparison to Traditional Development

### Traditional Approach
```
Problem → Human Analysis → Human Coding → Testing → Deployment
Timeline: Weeks to months
Iterations: 5-10 major versions
```

### Adaptive Evolution Approach
```
Problem → LLM Analysis → LLM Coding → Auto-Test → Auto-Deploy
Timeline: Hours to days
Iterations: 20-50+ micro-evolutions
```

## Future Enhancements

1. **Multi-Objective Optimization**
   - Balance bitrate vs quality vs speed
   - Pareto frontier tracking

2. **Ensemble Codecs**
   - Keep top 3 implementations
   - Route different video types to best codec

3. **Reinforcement Learning**
   - Reward good evolutions
   - Learn what code patterns work

4. **Cross-Experiment Learning**
   - Share learnings between codec, upscaling, etc.
   - Transfer learning for new domains

## Conclusion

The Adaptive Evolution system transforms the AI Video Codec from a static implementation into a **self-improving system**. The LLM doesn't just suggest improvements - it implements them, tests them, and deploys them automatically.

This is the future of autonomous AI development: systems that can identify their own flaws and rewrite themselves to fix them.

