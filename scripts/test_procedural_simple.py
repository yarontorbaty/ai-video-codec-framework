#!/usr/bin/env python3
"""
Simple test for procedural generation to isolate the bug
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_procedural_generation():
    """Test procedural generation step by step."""
    print("Testing procedural generation step by step...")
    
    try:
        from agents.procedural_generator import ProceduralCompressionAgent, SceneParameters
        
        print("✅ Imports successful")
        
        # Create agent
        agent = ProceduralCompressionAgent(resolution=(640, 480))  # Smaller resolution
        print("✅ Agent created")
        
        # Test scene parameters
        params = SceneParameters(
            time=0.0,
            resolution=(640, 480),
            complexity=0.5,
            color_palette='vibrant',
            motion_intensity=1.0,
            geometric_scale=1.0
        )
        print("✅ Scene parameters created")
        
        # Test scene generation
        frame = agent.generator.generate_scene(params)
        print(f"✅ Scene generated: {frame.shape}, dtype: {frame.dtype}")
        
        # Test video generation
        result = agent.generate_procedural_video("/tmp/simple_test.mp4", duration=2.0, fps=15.0)
        print(f"✅ Video generated: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_procedural_generation()
    if success:
        print("🎉 Procedural generation test PASSED!")
        sys.exit(0)
    else:
        print("❌ Procedural generation test FAILED!")
        sys.exit(1)

