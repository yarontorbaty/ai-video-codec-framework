#!/usr/bin/env python3
"""
Fix JSON Parsing
Extract the full neural codec code from the malformed JSON response.
"""

import json
import re

def fix_json_parsing():
    """Fix the JSON parsing to extract the full code."""
    print("🔧 Fixing JSON parsing to extract full neural codec code...")
    
    # Read the raw response
    with open('/tmp/raw_llm_response.txt', 'r') as f:
        response_text = f.read()
    
    print(f"📄 Raw response: {len(response_text)} characters")
    
    # Extract JSON from markdown
    if "```json" in response_text:
        json_str = response_text.split("```json")[1].split("```")[0].strip()
        print("✅ Extracted JSON from markdown code block")
    else:
        json_str = response_text.strip()
    
    print(f"📄 JSON string: {len(json_str)} characters")
    
    # The issue is that the code field contains unescaped quotes and newlines
    # Let's manually extract the code field
    
    # Find the start of the code field
    code_start_pattern = r'"generated_code"\s*:\s*\{[^}]*"code"\s*:\s*"'
    code_start_match = re.search(code_start_pattern, json_str)
    
    if not code_start_match:
        print("❌ Could not find code field start")
        return None
    
    start_pos = code_start_match.end()
    print(f"📍 Code starts at position: {start_pos}")
    
    # Find the end by looking for the closing quote before the next field
    # The code ends with a quote followed by a comma and "description"
    end_pattern = r'",\s*"description"\s*:'
    end_match = re.search(end_pattern, json_str[start_pos:])
    
    if not end_match:
        print("❌ Could not find code field end")
        return None
    
    end_pos = start_pos + end_match.start()
    print(f"📍 Code ends at position: {end_pos}")
    
    # Extract the raw code
    raw_code = json_str[start_pos:end_pos]
    print(f"📄 Raw code: {len(raw_code)} characters")
    
    # Unescape the code
    code = raw_code.replace('\\"', '"').replace('\\n', '\n')
    print(f"📄 Unescaped code: {len(code)} characters")
    
    # Save the extracted code
    with open('/tmp/extracted_neural_codec.py', 'w') as f:
        f.write(code)
    
    print("💾 Extracted code saved to /tmp/extracted_neural_codec.py")
    
    # Create a fixed JSON by replacing the malformed code with a placeholder
    # and then we'll add the real code separately
    
    # Replace the malformed code field with a placeholder
    fixed_json_str = json_str[:start_pos] + '"PLACEHOLDER_CODE_HERE"' + json_str[end_pos:]
    
    try:
        # Parse the fixed JSON
        analysis = json.loads(fixed_json_str)
        print("✅ Successfully parsed fixed JSON")
        
        # Replace the placeholder with the real code
        analysis['generated_code']['code'] = code
        
        # Convert to v2 format
        analysis['encoding_agent_code'] = code
        analysis['decoding_agent_code'] = code
        
        print(f"✅ Extracted full neural codec code: {len(code)} characters")
        print(f"   Hypothesis: {analysis.get('hypothesis', 'N/A')[:100]}...")
        print(f"   Expected bitrate: {analysis.get('expected_bitrate_mbps', 'N/A')} Mbps")
        print(f"   Expected PSNR: {analysis.get('expected_psnr_db', 'N/A')} dB")
        
        return analysis
        
    except json.JSONDecodeError as e:
        print(f"❌ Still can't parse JSON: {e}")
        return None

def test_extracted_code():
    """Test the extracted code."""
    print("\n🧪 Testing extracted neural codec code...")
    
    try:
        # Import the extracted code
        import sys
        sys.path.append('/tmp')
        from extracted_neural_codec import compress_video_frame, decompress_video_frame
        print("✅ Successfully imported extracted functions")
        
        # Test with a simple frame
        import numpy as np
        test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        config = {'keyframe_interval': 10, 'spatial_downscale': 0.5, 'jpeg_quality': 75}
        
        print("🧪 Testing compression...")
        compressed = compress_video_frame(test_frame, 0, config)
        print(f"✅ Compression successful: {len(compressed)} bytes")
        
        print("🧪 Testing decompression...")
        reconstructed = decompress_video_frame(compressed, 0, config)
        print(f"✅ Decompression successful: {reconstructed.shape}")
        
        # Calculate PSNR
        mse = np.mean((test_frame.astype(float) - reconstructed.astype(float)) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255**2 / mse)
            print(f"✅ PSNR: {psnr:.2f} dB")
        
        return True
        
    except Exception as e:
        print(f"❌ Code test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # Fix the JSON parsing
    analysis = fix_json_parsing()
    
    if analysis:
        # Test the extracted code
        success = test_extracted_code()
        
        if success:
            print("\n🎉 SUCCESS! Neural codec code extracted and working!")
            print("   The LLM is generating real, functional neural codec code")
            print("   The issue is just in the JSON parsing, not the code generation")
        else:
            print("\n❌ Code extraction succeeded but code execution failed")
    else:
        print("\n❌ Failed to extract code from JSON")
