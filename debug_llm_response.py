#!/usr/bin/env python3
"""
Debug LLM Response
Check what the LLM is actually returning and why code extraction fails.
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.llm_experiment_planner import LLMExperimentPlanner

def debug_llm_response():
    """Debug what the LLM is actually returning."""
    print("🔍 Debugging LLM Response")
    print("=" * 50)
    
    planner = LLMExperimentPlanner()
    
    # Get raw LLM response
    print("📡 Getting raw LLM response...")
    experiments = planner.analyze_recent_experiments(limit=2)
    
    if not experiments:
        print("❌ No experiments found")
        return
    
    # Call the LLM directly
    print("🤖 Calling LLM directly...")
    try:
        # Use the same method the planner uses
        prompt = planner.generate_analysis_prompt(experiments)
        
        # Get the raw response
        response = planner.client.messages.create(
            model=planner.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        print(f"✅ LLM Response received ({len(response_text)} chars)")
        
        # Save to file for inspection
        with open('/tmp/raw_llm_response.txt', 'w') as f:
            f.write(response_text)
        
        print("📝 Raw response saved to /tmp/raw_llm_response.txt")
        
        # Try to parse it
        print("\n🔍 Analyzing response structure...")
        
        # Check for JSON structure
        if response_text.strip().startswith('{'):
            print("✅ Response starts with JSON")
        else:
            print("⚠️  Response doesn't start with JSON")
        
        # Check for code blocks
        if '```python' in response_text:
            print("✅ Contains Python code blocks")
            code_blocks = response_text.split('```python')
            print(f"   Found {len(code_blocks) - 1} Python code blocks")
            for i, block in enumerate(code_blocks[1:], 1):
                code_content = block.split('```')[0]
                print(f"   Block {i}: {len(code_content)} chars")
                if len(code_content) > 100:
                    print(f"      Preview: {code_content[:100]}...")
        else:
            print("❌ No Python code blocks found")
        
        # Check for generated_code field
        if '"generated_code"' in response_text:
            print("✅ Contains generated_code field")
        else:
            print("❌ No generated_code field found")
        
        # Try the parsing method
        print("\n🧪 Testing parsing method...")
        parsed = planner._parse_llm_response(response_text)
        
        if parsed:
            print("✅ Parsing succeeded")
            print(f"   Hypothesis: {parsed.get('hypothesis', 'N/A')}")
            print(f"   Encoding code length: {len(parsed.get('encoding_agent_code', ''))}")
            print(f"   Decoding code length: {len(parsed.get('decoding_agent_code', ''))}")
        else:
            print("❌ Parsing failed")
        
        # Show first 500 chars of response
        print(f"\n📄 Response preview (first 500 chars):")
        print("-" * 50)
        print(response_text[:500])
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error calling LLM: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    debug_llm_response()
