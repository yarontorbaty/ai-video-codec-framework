#!/usr/bin/env python3
"""Quick parameter-supervised training with fewer samples for faster results."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_param_supervised import train_with_parameter_supervision

# Quick training: 2K samples, 15 epochs (~5-7 minutes)
print("\n🚀 QUICK TRAINING: 2K samples, 15 epochs (~5-7 minutes)\n")
model = train_with_parameter_supervision(num_samples=2000, num_epochs=15)

