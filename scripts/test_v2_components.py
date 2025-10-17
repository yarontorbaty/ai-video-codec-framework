#!/usr/bin/env python3
"""
Component Test Script for v2.0 Neural Codec
Tests each component independently to verify they work
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("🔍 Testing Imports")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test core Python modules
    try:
        import numpy as np
        print("✅ NumPy imported")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        tests_failed += 1
    
    try:
        import cv2
        print("✅ OpenCV imported")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        tests_failed += 1
    
    try:
        import boto3
        print("✅ Boto3 imported")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ Boto3 import failed: {e}")
        tests_failed += 1
    
    try:
        import torch
        print(f"✅ PyTorch imported (version: {torch.__version__})")
        if torch.cuda.is_available():
            print(f"   🎮 CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"   ⚠️  CUDA not available (CPU only)")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        tests_failed += 1
    
    print(f"\nImports: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def test_encoding_agent():
    """Test EncodingAgent can be imported and instantiated."""
    print("\n" + "=" * 60)
    print("🔍 Testing EncodingAgent")
    print("=" * 60)
    
    try:
        from src.agents.encoding_agent import (
            EncodingAgent,
            SceneClassifier,
            IFrameVAE,
            SemanticDescriptionGenerator,
            CompressionStrategySelector
        )
        print("✅ EncodingAgent imports successful")
        
        # Test instantiation
        config = {
            'latent_dim': 512,
            'description_dim': 256,
            'i_frame_interval': 30,
            'target_bitrate_mbps': 1.0,
            'target_quality_ssim': 0.95
        }
        
        encoder = EncodingAgent(config)
        print("✅ EncodingAgent instantiated")
        
        # Test components
        classifier = SceneClassifier()
        print("✅ SceneClassifier instantiated")
        
        vae = IFrameVAE(latent_dim=512)
        print("✅ IFrameVAE instantiated")
        
        semantic = SemanticDescriptionGenerator(latent_dim=512, description_dim=256)
        print("✅ SemanticDescriptionGenerator instantiated")
        
        selector = CompressionStrategySelector()
        print("✅ CompressionStrategySelector instantiated")
        
        # Test strategy selection
        scene_info = {
            'scene_type': 'talking_head',
            'complexity': 0.5,
            'motion_intensity': 0.3
        }
        strategy = selector.select_strategy(scene_info, config)
        print(f"✅ Strategy selection works: {strategy}")
        
        return True
        
    except Exception as e:
        print(f"❌ EncodingAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_decoding_agent():
    """Test DecodingAgent can be imported and instantiated."""
    print("\n" + "=" * 60)
    print("🔍 Testing DecodingAgent")
    print("=" * 60)
    
    try:
        from src.agents.decoding_agent import (
            DecodingAgent,
            LightweightIFrameDecoder,
            LightweightVideoGenerator,
            TemporalConsistencyEnhancer
        )
        print("✅ DecodingAgent imports successful")
        
        # Test instantiation
        config = {
            'latent_dim': 512,
            'description_dim': 256,
            'use_temporal_enhancement': True
        }
        
        decoder = DecodingAgent(config)
        print("✅ DecodingAgent instantiated")
        
        # Test components
        i_frame_decoder = LightweightIFrameDecoder(latent_dim=512)
        print("✅ LightweightIFrameDecoder instantiated")
        
        video_gen = LightweightVideoGenerator(description_dim=256)
        print("✅ LightweightVideoGenerator instantiated")
        
        enhancer = TemporalConsistencyEnhancer()
        print("✅ TemporalConsistencyEnhancer instantiated")
        
        return True
        
    except Exception as e:
        print(f"❌ DecodingAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_orchestrator():
    """Test GPU-first orchestrator can be imported."""
    print("\n" + "=" * 60)
    print("🔍 Testing GPU-First Orchestrator")
    print("=" * 60)
    
    try:
        from src.agents.gpu_first_orchestrator import (
            GPUFirstOrchestrator,
            ExperimentPhase
        )
        print("✅ GPUFirstOrchestrator imports successful")
        
        # Test enum
        phases = [p.value for p in ExperimentPhase]
        print(f"✅ Experiment phases: {', '.join(phases)}")
        
        # Note: Don't instantiate orchestrator (requires AWS)
        print("✅ Orchestrator structure verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_worker():
    """Test GPU worker can be imported."""
    print("\n" + "=" * 60)
    print("🔍 Testing GPU Worker")
    print("=" * 60)
    
    try:
        from workers.neural_codec_gpu_worker import (
            NeuralCodecExecutor,
            NeuralCodecWorker
        )
        print("✅ GPU Worker imports successful")
        
        # Note: Don't instantiate worker (requires GPU)
        print("✅ GPU Worker structure verified")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU Worker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aws_connectivity():
    """Test AWS connectivity."""
    print("\n" + "=" * 60)
    print("🔍 Testing AWS Connectivity")
    print("=" * 60)
    
    try:
        import boto3
        
        # Test STS (identity)
        try:
            sts = boto3.client('sts', region_name='us-east-1')
            identity = sts.get_caller_identity()
            print(f"✅ AWS Identity: {identity['Account']}")
        except Exception as e:
            print(f"❌ AWS credentials not configured: {e}")
            return False
        
        # Test SQS
        try:
            sqs = boto3.client('sqs', region_name='us-east-1')
            queue_url = 'https://sqs.us-east-1.amazonaws.com/580473065386/ai-video-codec-training-queue'
            response = sqs.get_queue_attributes(
                QueueUrl=queue_url,
                AttributeNames=['ApproximateNumberOfMessages']
            )
            msg_count = response['Attributes']['ApproximateNumberOfMessages']
            print(f"✅ SQS Queue accessible (messages: {msg_count})")
        except Exception as e:
            print(f"⚠️  SQS Queue not accessible: {e}")
        
        # Test DynamoDB
        try:
            dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
            table = dynamodb.Table('ai-video-codec-experiments')
            table.load()
            print(f"✅ DynamoDB Table accessible")
        except Exception as e:
            print(f"⚠️  DynamoDB Table not accessible: {e}")
        
        # Test S3
        try:
            s3 = boto3.client('s3', region_name='us-east-1')
            account_id = identity['Account']
            bucket = f'ai-video-codec-videos-{account_id}'
            s3.list_objects_v2(Bucket=bucket, MaxKeys=1)
            print(f"✅ S3 Bucket accessible: {bucket}")
        except Exception as e:
            print(f"⚠️  S3 Bucket not accessible: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ AWS connectivity test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 v2.0 Component Test Suite")
    print("=" * 60)
    print()
    
    results = {
        'Imports': test_imports(),
        'EncodingAgent': test_encoding_agent(),
        'DecodingAgent': test_decoding_agent(),
        'Orchestrator': test_orchestrator(),
        'GPU Worker': test_gpu_worker(),
        'AWS Connectivity': test_aws_connectivity()
    }
    
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Total: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All tests passed! System is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Fix issues before deploying.")
        return 1


if __name__ == '__main__':
    sys.exit(main())


