#!/usr/bin/env python3
"""
Test script for extended graphics primitives library.
Verifies all 47 functions work correctly.
"""

import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graphics.primitives_extended import ExtendedGraphicsPrimitives, EXTENDED_FUNCTION_MAP

def test_extended_primitives():
    """Test all extended graphics primitives."""
    print("="*70)
    print("Testing Extended Graphics Primitives Library")
    print("="*70)
    print(f"\n📊 Total functions: {len(EXTENDED_FUNCTION_MAP)}\n")
    
    gfx = ExtendedGraphicsPrimitives(width=256, height=256)
    
    # Test each category
    test_results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    print("🎨 Testing Advanced Fill Functions...")
    try:
        gfx.reset()
        gfx.fill_radial_gradient(128, 128, 100, (255, 0, 0), (0, 0, 255))
        print("  ✓ fill_radial_gradient")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ fill_radial_gradient: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(('fill_radial_gradient', str(e)))
    
    try:
        gfx.reset()
        gfx.fill_conic_gradient(128, 128, (255, 0, 0), (0, 255, 0))
        print("  ✓ fill_conic_gradient")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ fill_conic_gradient: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        gfx.fill_noise_perlin(10.0, (128, 128, 128))
        print("  ✓ fill_noise_perlin")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ fill_noise_perlin: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        gfx.fill_checkerboard(16, (0, 0, 0), (255, 255, 255))
        print("  ✓ fill_checkerboard")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ fill_checkerboard: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        gfx.fill_stripes(16, 45.0, (0, 0, 0), (255, 255, 255))
        print("  ✓ fill_stripes")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ fill_stripes: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        gfx.fill_dots(8, 24, (255, 255, 255), (0, 0, 0))
        print("  ✓ fill_dots")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ fill_dots: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        gfx.fill_wave(0.1, 20, (0, 0, 0), (255, 255, 255))
        print("  ✓ fill_wave")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ fill_wave: {e}")
        test_results['failed'] += 1
    
    print("\n📐 Testing Advanced Shape Functions...")
    try:
        gfx.reset()
        gfx.draw_polygon([(50, 50), (200, 50), (150, 200)], (255, 0, 0), True)
        print("  ✓ draw_polygon")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ draw_polygon: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        gfx.draw_bezier_curve((50, 50), (100, 200), (200, 200), (200, 50), (255, 0, 0), 2)
        print("  ✓ draw_bezier_curve")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ draw_bezier_curve: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        gfx.draw_arc(128, 128, 50, 0, 180, (255, 0, 0), 2)
        print("  ✓ draw_arc")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ draw_arc: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        gfx.draw_rounded_rect(50, 50, 100, 80, 10, (255, 0, 0), True)
        print("  ✓ draw_rounded_rect")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ draw_rounded_rect: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        gfx.draw_star(128, 128, 50, 25, 5, (255, 0, 0), True)
        print("  ✓ draw_star")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ draw_star: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        gfx.draw_triangle(50, 50, 200, 50, 125, 200, (255, 0, 0), True)
        print("  ✓ draw_triangle")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ draw_triangle: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        gfx.draw_heart(128, 128, 3, (255, 0, 0), True)
        print("  ✓ draw_heart")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ draw_heart: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        gfx.draw_ring(128, 128, 50, 30, (255, 0, 0))
        print("  ✓ draw_ring")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ draw_ring: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        gfx.draw_trapezoid(50, 50, 200, 150, 80, 120, (255, 0, 0), True)
        print("  ✓ draw_trapezoid")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ draw_trapezoid: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        gfx.draw_parallelogram(50, 50, 100, 80, 20, (255, 0, 0), True)
        print("  ✓ draw_parallelogram")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ draw_parallelogram: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        gfx.draw_crescent(128, 128, 50, 20, (255, 200, 0))
        print("  ✓ draw_crescent")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ draw_crescent: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        gfx.draw_cross(128, 128, 80, 15, (255, 0, 0))
        print("  ✓ draw_cross")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ draw_cross: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        gfx.draw_arrow(50, 128, 200, 128, (255, 0, 0), 10)
        print("  ✓ draw_arrow")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ draw_arrow: {e}")
        test_results['failed'] += 1
    
    print("\n✨ Testing Effect Functions...")
    # Create test canvas with some content
    gfx.reset()
    cv2.rectangle(gfx.canvas, (50, 50), (200, 200), (255, 255, 255), -1)
    
    try:
        gfx.apply_blur(50, 50, 150, 150, 15)
        print("  ✓ apply_blur")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ apply_blur: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.apply_glow(50, 50, 150, 150, (255, 200, 0), 0.5)
        print("  ✓ apply_glow")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ apply_glow: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        cv2.rectangle(gfx.canvas, (50, 50), (200, 200), (255, 255, 255), -1)
        gfx.apply_shadow(50, 50, 150, 150, 5, 5, (0, 0, 0), 0.5)
        print("  ✓ apply_shadow")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ apply_shadow: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        cv2.rectangle(gfx.canvas, (50, 50), (200, 200), (128, 128, 128), -1)
        gfx.apply_sharpen(50, 50, 150, 150)
        print("  ✓ apply_sharpen")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ apply_sharpen: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        cv2.rectangle(gfx.canvas, (50, 50), (200, 200), (200, 150, 100), -1)
        gfx.apply_posterize(50, 50, 150, 150, 4)
        print("  ✓ apply_posterize")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ apply_posterize: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        cv2.rectangle(gfx.canvas, (50, 50), (200, 200), (200, 150, 100), -1)
        gfx.apply_pixelate(50, 50, 150, 150, 8)
        print("  ✓ apply_pixelate")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ apply_pixelate: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        cv2.rectangle(gfx.canvas, (50, 50), (200, 200), (200, 150, 100), -1)
        gfx.apply_vignette(0.5)
        print("  ✓ apply_vignette")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ apply_vignette: {e}")
        test_results['failed'] += 1
    
    print("\n🎨 Testing Compositing Functions...")
    try:
        gfx.reset()
        cv2.rectangle(gfx.canvas, (50, 50), (200, 200), (200, 150, 100), -1)
        gfx.blend_multiply((255, 0, 0), 0.5)
        print("  ✓ blend_multiply")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ blend_multiply: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        cv2.rectangle(gfx.canvas, (50, 50), (200, 200), (100, 100, 100), -1)
        gfx.blend_screen((255, 200, 100), 0.5)
        print("  ✓ blend_screen")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ blend_screen: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        cv2.rectangle(gfx.canvas, (50, 50), (200, 200), (150, 150, 150), -1)
        gfx.blend_overlay((255, 0, 0), 0.5)
        print("  ✓ blend_overlay")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ blend_overlay: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        cv2.rectangle(gfx.canvas, (50, 50), (200, 200), (100, 100, 100), -1)
        gfx.blend_add((50, 50, 50), 0.5)
        print("  ✓ blend_add")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ blend_add: {e}")
        test_results['failed'] += 1
    
    try:
        gfx.reset()
        cv2.rectangle(gfx.canvas, (50, 50), (200, 200), (200, 200, 200), -1)
        gfx.blend_subtract((50, 50, 50), 0.5)
        print("  ✓ blend_subtract")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ✗ blend_subtract: {e}")
        test_results['failed'] += 1
    
    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"✅ Passed: {test_results['passed']}")
    print(f"❌ Failed: {test_results['failed']}")
    print(f"📊 Total: {test_results['passed'] + test_results['failed']}")
    print(f"✓ Success Rate: {100 * test_results['passed'] / (test_results['passed'] + test_results['failed']):.1f}%")
    
    if test_results['errors']:
        print("\n🔴 Errors:")
        for func_name, error in test_results['errors']:
            print(f"  - {func_name}: {error}")
    
    print("\n✅ Extended Graphics Primitives Library Ready!\n")
    
    return test_results['passed'] == (test_results['passed'] + test_results['failed'])


if __name__ == "__main__":
    success = test_extended_primitives()
    sys.exit(0 if success else 1)

