#!/usr/bin/env python3
"""
Test LLM Code Generation
Verify that the LLM is generating actual neural codec code.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.llm_experiment_planner import LLMExperimentPlanner

def test_llm_code_generation():
    """Test if LLM generates actual code."""
    print("🧪 Testing LLM Code Generation")
    print("=" * 50)
    
    planner = LLMExperimentPlanner()
    
    # Generate experiment
    print("📡 Requesting LLM to generate neural codec experiment...")
    analysis = planner.plan_next_experiment()
    
    if not analysis:
        print("❌ LLM failed to generate analysis")
        return False
    
    print(f"✅ LLM analysis generated")
    print(f"   Hypothesis: {analysis.get('hypothesis', 'N/A')}")
    print(f"   Strategy: {analysis.get('compression_strategy', 'N/A')}")
    
    # Check encoding code
    encoding_code = analysis.get('encoding_agent_code', '')
    print(f"   Encoding code length: {len(encoding_code)} characters")
    
    if len(encoding_code) > 0:
        print("✅ Encoding code generated")
        print("   Preview:")
        print("   " + encoding_code[:200].replace('\n', '\n   ') + "...")
    else:
        print("❌ No encoding code generated")
        return False
    
    # Check decoding code
    decoding_code = analysis.get('decoding_agent_code', '')
    print(f"   Decoding code length: {len(decoding_code)} characters")
    
    if len(decoding_code) > 0:
        print("✅ Decoding code generated")
        print("   Preview:")
        print("   " + decoding_code[:200].replace('\n', '\n   ') + "...")
    else:
        print("❌ No decoding code generated")
        return False
    
    # Test code execution
    print("\n🧪 Testing code execution...")
    
    try:
        from src.utils.code_sandbox import CodeSandbox
        sandbox = CodeSandbox()
        
        # Test encoding code
        print("   Testing encoding agent...")
        encoding_result = sandbox.execute_function(
            encoding_code,
            'run_encoding_agent',
            {'device': 'cpu'}
        )
        
        if encoding_result[0]:  # Success
            print(f"   ✅ Encoding agent executed successfully")
            print(f"   Result: {encoding_result[1]}")
        else:
            print(f"   ❌ Encoding agent failed: {encoding_result[2]}")
            return False
        
        # Test decoding code
        print("   Testing decoding agent...")
        decoding_result = sandbox.execute_function(
            decoding_code,
            'run_decoding_agent',
            {'device': 'cpu', 'encoding_data': encoding_result[1]}
        )
        
        if decoding_result[0]:  # Success
            print(f"   ✅ Decoding agent executed successfully")
            print(f"   Result: {decoding_result[1]}")
        else:
            print(f"   ❌ Decoding agent failed: {decoding_result[2]}")
            return False
        
        print("\n🎉 LLM Code Generation Test PASSED!")
        print("   Both encoding and decoding agents generated and executed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Code execution test failed: {e}")
        return False

if __name__ == '__main__':
    success = test_llm_code_generation()
    sys.exit(0 if success else 1)
