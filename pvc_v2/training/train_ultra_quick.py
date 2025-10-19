#!/usr/bin/env python3
"""Super quick parameter-supervised training - proof of concept."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_param_supervised import train_with_parameter_supervision

# Ultra-quick training: 500 samples, 10 epochs (~2-3 minutes)
print("\n⚡ ULTRA-QUICK TRAINING: 500 samples, 10 epochs (~2-3 minutes)\n")
model = train_with_parameter_supervision(num_samples=500, num_epochs=10)

