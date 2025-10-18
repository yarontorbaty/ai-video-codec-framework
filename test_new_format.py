#!/usr/bin/env python3
"""
Test New Format
Force the LLM to use the new sectioned format.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.llm_experiment_planner import LLMExperimentPlanner

def test_new_format():
    """Test the new format by being explicit about it."""
    print("🧪 Testing new format with explicit instructions...")
    
    planner = LLMExperimentPlanner()
    
    # Create a simple prompt that forces the new format
    simple_prompt = """
You are an AI video codec expert. Analyze these experiment results and provide your response in the EXACT format below:

## ANALYSIS
```json
{
  "root_cause": "Why compression is failing",
  "insights": ["Key patterns observed"],
  "hypothesis": "What changes would improve results",
  "expected_bitrate_mbps": 4.2,
  "expected_psnr_db": 32.0,
  "confidence_score": 0.75
}
```

## NEURAL CODEC IMPLEMENTATION
```python
# Your complete neural codec implementation here
# Include BOTH compress_video_frame() AND decompress_video_frame() functions

import numpy as np
import cv2
import struct

def compress_video_frame(frame: np.ndarray, frame_index: int, config: dict) -> bytes:
    # Simple test implementation
    h, w = frame.shape[:2]
    # Downsample by 50%
    small = cv2.resize(frame, (w//2, h//2))
    # JPEG compress
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
    _, encoded = cv2.imencode('.jpg', small, encode_param)
    return encoded.tobytes()

def decompress_video_frame(compressed_data: bytes, frame_index: int, config: dict) -> np.ndarray:
    # Simple test implementation
    nparr = np.frombuffer(compressed_data, np.uint8)
    decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Upsample back to original size (assuming 1920x1080)
    upscaled = cv2.resize(decoded, (1920, 1080))
    return upscaled
```

## DESCRIPTION
Simple neural codec using downsampling + JPEG compression for testing.

Recent experiments show issues with procedural generation. Please analyze and respond in the EXACT format above.
"""

    try:
        # Call LLM directly with the simple prompt
        response = planner.client.messages.create(
            model=planner.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": simple_prompt}]
        )
        
        response_text = response.content[0].text
        print(f"✅ LLM Response received ({len(response_text)} chars)")
        
        # Save response for inspection
        with open('/tmp/new_format_response.txt', 'w') as f:
            f.write(response_text)
        
        print("📝 Response saved to /tmp/new_format_response.txt")
        
        # Test parsing
        parsed = planner._parse_llm_response(response_text)
        
        if parsed:
            print("✅ Parsing succeeded")
            print(f"   Hypothesis: {parsed.get('hypothesis', 'N/A')}")
            print(f"   Encoding code length: {len(parsed.get('encoding_agent_code', ''))}")
            print(f"   Decoding code length: {len(parsed.get('decoding_agent_code', ''))}")
            
            if len(parsed.get('encoding_agent_code', '')) > 100:
                print("🎉 SUCCESS! New format working with real code!")
                return True
            else:
                print("❌ Code extraction failed")
                return False
        else:
            print("❌ Parsing failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    success = test_new_format()
    if success:
        print("\n🎉 New format is working! The LLM can now generate code in a parseable format.")
    else:
        print("\n❌ New format test failed")
