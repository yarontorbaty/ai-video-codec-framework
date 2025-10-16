#!/usr/bin/env python3
"""
Test PyTorch and OpenCV installation
"""

import sys
import os

print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ PyTorch location: {torch.__file__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

try:
    import cv2
    print(f"✅ OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")

try:
    import numpy as np
    print(f"✅ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

# Test basic functionality
try:
    import cv2
    import numpy as np
    
    # Create a test frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite("/tmp/test_frame.jpg", frame)
    
    # Check file size
    file_size = os.path.getsize("/tmp/test_frame.jpg")
    print(f"✅ Test frame created: /tmp/test_frame.jpg ({file_size} bytes)")
    
except Exception as e:
    print(f"❌ Test frame creation failed: {e}")

print("Test completed!")
