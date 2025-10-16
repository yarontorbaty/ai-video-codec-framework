#!/usr/bin/env python3
"""
Debug procedural generation to find the exact tensor conversion issue
"""

import sys
import os
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def debug_procedural_generation():
    """Debug procedural generation step by step."""
    print("🔍 Debugging procedural generation...")
    
    try:
        from agents.procedural_generator import ProceduralCompressionAgent, SceneParameters
        
        print("✅ Imports successful")
        
        # Create agent with same parameters as real experiment
        agent = ProceduralCompressionAgent(resolution=(1920, 1080))
        print("✅ Agent created with 1920x1080 resolution")
        
        # Test the exact same call as in real_experiment.py
        print("🎬 Testing generate_procedural_video with 10s duration, 30fps...")
        results = agent.generate_procedural_video(
            "/tmp/procedural_test.mp4", 
            duration=10.0, 
            fps=30.0
        )
        
        print(f"✅ Procedural video generated successfully: {results}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_procedural_generation()
    if success:
        print("🎉 Procedural generation debug PASSED!")
        sys.exit(0)
    else:
        print("❌ Procedural generation debug FAILED!")
        sys.exit(1)
