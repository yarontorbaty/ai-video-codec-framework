#!/usr/bin/env python3
"""
Test AI Codec Agents
Quick test to verify the AI agents are working correctly.
"""

import sys
import os
import logging
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from agents.ai_codec_agent import AICodecAgent
        print("✓ AICodecAgent imported successfully")
    except Exception as e:
        print(f"✗ Failed to import AICodecAgent: {e}")
        return False
    
    try:
        from agents.procedural_generator import ProceduralCompressionAgent
        print("✓ ProceduralCompressionAgent imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ProceduralCompressionAgent: {e}")
        return False
    
    try:
        from agents.experiment_orchestrator import ExperimentOrchestrator
        print("✓ ExperimentOrchestrator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ExperimentOrchestrator: {e}")
        return False
    
    try:
        from utils.aws_utils import AWSUtils
        print("✓ AWSUtils imported successfully")
    except Exception as e:
        print(f"✗ Failed to import AWSUtils: {e}")
        return False
    
    try:
        from utils.metrics import MetricsCalculator
        print("✓ MetricsCalculator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import MetricsCalculator: {e}")
        return False
    
    try:
        from utils.video_utils import VideoProcessor
        print("✓ VideoProcessor imported successfully")
    except Exception as e:
        print(f"✗ Failed to import VideoProcessor: {e}")
        return False
    
    return True

def test_neural_networks():
    """Test neural network creation."""
    print("\nTesting neural networks...")
    
    try:
        from agents.ai_codec_agent import SemanticEncoder, MotionPredictor, GenerativeRefiner
        
        # Test SemanticEncoder
        encoder = SemanticEncoder()
        test_input = torch.randn(1, 3, 64, 64)
        output = encoder(test_input)
        print(f"✓ SemanticEncoder: input {test_input.shape} -> output {output.shape}")
        
        # Test MotionPredictor
        motion_predictor = MotionPredictor()
        current_frame = torch.randn(1, 3, 64, 64)
        reference_frame = torch.randn(1, 3, 64, 64)
        motion_output = motion_predictor(current_frame, reference_frame)
        print(f"✓ MotionPredictor: input {current_frame.shape} -> output {motion_output.shape}")
        
        # Test GenerativeRefiner
        refiner = GenerativeRefiner()
        refiner_input = torch.randn(1, 3, 64, 64)
        refiner_output = refiner(refiner_input)
        print(f"✓ GenerativeRefiner: input {refiner_input.shape} -> output {refiner_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Neural network test failed: {e}")
        return False

def test_procedural_generation():
    """Test procedural generation."""
    print("\nTesting procedural generation...")
    
    try:
        from agents.procedural_generator import ProceduralSceneGenerator, SceneParameters
        
        # Create generator
        generator = ProceduralSceneGenerator(resolution=(320, 240))  # Small resolution for testing
        
        # Test scene parameters
        params = SceneParameters(
            time=0.0,
            resolution=(320, 240),
            complexity=0.5,
            color_palette='vibrant',
            motion_intensity=1.0,
            geometric_scale=1.0
        )
        
        # Test geometric pattern generation
        geometric_frame = generator.generate_geometric_pattern(params)
        print(f"✓ Geometric pattern: {geometric_frame.shape}")
        
        # Test fractal texture generation
        fractal_frame = generator.generate_fractal_texture(params)
        print(f"✓ Fractal texture: {fractal_frame.shape}")
        
        # Test plasma effect
        plasma_frame = generator.generate_plasma_effect(params)
        print(f"✓ Plasma effect: {plasma_frame.shape}")
        
        # Test complete scene generation
        scene_frame = generator.generate_scene(params)
        print(f"✓ Complete scene: {scene_frame.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Procedural generation test failed: {e}")
        return False

def test_metrics_calculation():
    """Test metrics calculation."""
    print("\nTesting metrics calculation...")
    
    try:
        from utils.metrics import MetricsCalculator
        
        # Create test images
        test_image1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_image2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Save test images
        import cv2
        cv2.imwrite('test_image1.jpg', test_image1)
        cv2.imwrite('test_image2.jpg', test_image2)
        
        # Test metrics calculator
        metrics_calc = MetricsCalculator()
        
        # Test PSNR calculation
        psnr = metrics_calc.calculate_psnr('test_image1.jpg', 'test_image2.jpg')
        print(f"✓ PSNR calculation: {psnr:.2f} dB")
        
        # Test SSIM calculation
        ssim = metrics_calc.calculate_ssim('test_image1.jpg', 'test_image2.jpg')
        print(f"✓ SSIM calculation: {ssim:.4f}")
        
        # Test bitrate calculation
        bitrate = metrics_calc.calculate_bitrate('test_image1.jpg')
        print(f"✓ Bitrate calculation: {bitrate:.2f} Mbps")
        
        # Clean up
        os.remove('test_image1.jpg')
        os.remove('test_image2.jpg')
        
        return True
        
    except Exception as e:
        print(f"✗ Metrics calculation test failed: {e}")
        return False

def test_video_processing():
    """Test video processing utilities."""
    print("\nTesting video processing...")
    
    try:
        from utils.video_utils import VideoProcessor
        
        # Create a simple test video
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('test_video.mp4', fourcc, 30.0, (320, 240))
        
        # Write a few frames
        for i in range(30):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
        # Test video processor
        processor = VideoProcessor()
        
        # Test video info
        info = processor.get_video_info('test_video.mp4')
        print(f"✓ Video info: {info['width']}x{info['height']}, {info['fps']} fps")
        
        # Test frame extraction
        frames = processor.extract_frames('test_video.mp4', 'test_frames', max_frames=5)
        print(f"✓ Frame extraction: {len(frames)} frames extracted")
        
        # Test video statistics
        stats = processor.get_video_statistics('test_video.mp4')
        print(f"✓ Video statistics: {stats['file_size_mb']:.2f} MB, {stats['bitrate_mbps']:.2f} Mbps")
        
        # Clean up
        os.remove('test_video.mp4')
        import shutil
        shutil.rmtree('test_frames', ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"✗ Video processing test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("AI Video Codec Framework - Agent Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Neural Network Test", test_neural_networks),
        ("Procedural Generation Test", test_procedural_generation),
        ("Metrics Calculation Test", test_metrics_calculation),
        ("Video Processing Test", test_video_processing),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AI agents are ready to run.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
