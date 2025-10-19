#!/usr/bin/env python3
"""
Enhanced Dataset with Parameter Ground Truth

Includes both function IDs and normalized parameters for training.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from graphics.primitives import FunctionCall


class FunctionSequenceDatasetWithParams(Dataset):
    """Dataset with function IDs AND parameter ground truth."""
    
    def __init__(self, frames: List[np.ndarray], function_sequences: List[List[FunctionCall]],
                 max_seq_len: int = 20, end_token_id: int = 42):
        """
        Initialize dataset.
        
        Args:
            frames: List of rendered frames (uint8, H x W x 3)
            function_sequences: List of function call lists (with params!)
            max_seq_len: Maximum sequence length
            end_token_id: ID for END token (NUM_EXTENDED_FUNCTIONS)
        """
        self.frames = frames
        self.function_sequences = function_sequences
        self.max_seq_len = max_seq_len
        self.end_token_id = end_token_id
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            (frame_tensor, function_ids_tensor, parameters_tensor)
        """
        # Convert frame to tensor (3 x H x W), float32, [0, 1]
        frame = self.frames[idx]
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        # Extract function IDs
        func_calls = self.function_sequences[idx]
        
        # Handle both dict and FunctionCall objects
        if len(func_calls) > 0:
            if isinstance(func_calls[0], dict):
                # Dictionary format from ExtendedSyntheticGenerator
                func_ids = [f['func_id'] for f in func_calls]
            else:
                # FunctionCall objects
                func_ids = [f.func_id for f in func_calls]
        else:
            func_ids = []
        
        # Extract normalized parameters (10 values per function)
        # [coords(4), color1(3), color2(3)]
        params_list = []
        for call in func_calls:
            if isinstance(call, dict):
                # Dictionary format - get normalized_params directly
                params_array = call.get('normalized_params', np.zeros(10, dtype=np.float32))
            else:
                # FunctionCall object
                params_array = call.get_normalized_params(
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0]
                )
            params_list.append(params_array)
        
        # Pad sequences to max_seq_len
        while len(func_ids) < self.max_seq_len:
            func_ids.append(self.end_token_id)  # END token
            params_list.append(np.zeros(10, dtype=np.float32))  # Zero padding
        
        # Truncate if too long
        func_ids = func_ids[:self.max_seq_len]
        params_list = params_list[:self.max_seq_len]
        
        # Convert to tensors
        func_ids_tensor = torch.tensor(func_ids, dtype=torch.long)
        params_tensor = torch.from_numpy(np.stack(params_list))  # (seq_len, 10)
        
        return frame_tensor, func_ids_tensor, params_tensor


if __name__ == '__main__':
    """Test the enhanced dataset."""
    print("Testing enhanced dataset with parameters...")
    
    from training.synthetic_generator import SyntheticDataGenerator
    
    # Generate test data
    generator = SyntheticDataGenerator(256, 256)
    frames, function_sequences = generator.generate_dataset(num_samples=10, anime_ratio=0.5)
    
    # Create dataset
    dataset = FunctionSequenceDatasetWithParams(frames, function_sequences)
    
    # Test a sample
    frame, func_ids, params = dataset[0]
    
    print(f"\n✅ Dataset test:")
    print(f"   Frame shape: {frame.shape}")
    print(f"   Function IDs shape: {func_ids.shape}")
    print(f"   Parameters shape: {params.shape}")
    print(f"\n   Function IDs: {func_ids[:5]}")
    print(f"   First function params: {params[0]}")
    print(f"     Coords: {params[0, :4]}")
    print(f"     Color1: {params[0, 4:7]}")
    print(f"     Color2: {params[0, 7:10]}")
    print("\n✅ Enhanced dataset working!")

