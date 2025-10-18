#!/usr/bin/env python3
"""
Launch the first v2.0 Neural Codec experiment
"""

import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.gpu_first_orchestrator import GPUFirstOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Launch v2.0 experiment"""
    
    print("=" * 80)
    print("🚀 LAUNCHING v2.0 NEURAL CODEC EXPERIMENT")
    print("=" * 80)
    print()
    
    # Check API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ERROR: ANTHROPIC_API_KEY not set")
        print("Please load it from Secrets Manager:")
        print("  export ANTHROPIC_API_KEY=$(aws secretsmanager get-secret-value --secret-id ai-video-codec/anthropic-api-key --query SecretString --output text)")
        sys.exit(1)
    
    print(f"✅ API Key loaded (length: {len(api_key)})")
    print()
    
    # Initialize orchestrator
    print("🎯 Initializing GPU-First Orchestrator...")
    try:
        orchestrator = GPUFirstOrchestrator()
        print("✅ Orchestrator initialized")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run experiment cycle
    print("🎬 Starting Experiment Cycle...")
    print("   This will:")
    print("   1. Design neural codec architecture (LLM)")
    print("   2. Dispatch to GPU worker via SQS")
    print("   3. Wait for GPU execution")
    print("   4. Analyze results")
    print()
    
    try:
        result = orchestrator.run_experiment_cycle(iteration=1)
        
        print()
        print("=" * 80)
        
        if result.get('success'):
            print("✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print()
            print(f"📊 Experiment ID: {result['experiment_id']}")
            print(f"📈 Phase: {result['phase']}")
            
            metrics = result.get('metrics', {})
            print()
            print("Results:")
            print(f"  🎯 Target Achieved: {result.get('target_achieved', False)}")
            print(f"  💾 Bitrate: {metrics.get('bitrate_mbps', 0):.4f} Mbps")
            print(f"  📊 PSNR: {metrics.get('psnr_db', 0):.2f} dB")
            print(f"  👁️  SSIM: {metrics.get('ssim', 0):.4f}")
            print(f"  🔢 TOPS/frame: {metrics.get('tops_per_frame', 0):.2f}")
            print()
        else:
            print("❌ EXPERIMENT FAILED")
            print("=" * 80)
            print()
            print(f"Reason: {result.get('reason', 'unknown')}")
            print(f"Phase: {result.get('phase', 'unknown')}")
            print()
            print("Details:")
            print(result)
            
    except KeyboardInterrupt:
        print()
        print("⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

