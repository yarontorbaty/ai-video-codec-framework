#!/usr/bin/env python3
"""
Enhanced PVC v2.0 Model with Full Sequence Parameter Prediction

Key improvement: Predicts parameters for ALL functions in the sequence,
not just the first one.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.network import CNNEncoder, RNNDecoder


class SequenceParameterPredictor(nn.Module):
    """
    Predicts parameters for an entire sequence of functions.
    
    For each position in the sequence, predicts 10 normalized parameters:
    - coords (4): x1, y1, x2, y2 (normalized to [0, 1])
    - color1 (3): R, G, B (normalized to [0, 1])
    - color2 (3): R, G, B (normalized to [0, 1])
    """
    
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128, num_functions: int = 42):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_functions = num_functions
        
        # Function embedding (for conditioning parameter prediction)
        self.func_embedding = nn.Embedding(num_functions + 1, 64)  # +1 for END
        
        # RNN for sequential parameter prediction
        self.rnn = nn.GRU(
            input_size=feature_dim + 64,  # features + function embedding
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Parameter prediction head
        self.param_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 parameters per function
        )
    
    def forward(self, 
                features: torch.Tensor,
                func_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict parameters for sequence of functions.
        
        Args:
            features: Visual features (batch_size x feature_dim)
            func_ids: Function ID sequence (batch_size x seq_len)
            
        Returns:
            parameters: (batch_size x seq_len x 10)
                       Normalized parameters for each function
        """
        batch_size, seq_len = func_ids.shape
        
        # Embed function IDs
        func_emb = self.func_embedding(func_ids)  # (batch x seq_len x 64)
        
        # Expand features to match sequence length
        features_expanded = features.unsqueeze(1).expand(-1, seq_len, -1)  # (batch x seq_len x feature_dim)
        
        # Concatenate features and function embeddings
        rnn_input = torch.cat([features_expanded, func_emb], dim=2)  # (batch x seq_len x (feature_dim + 64))
        
        # Process through RNN
        rnn_output, _ = self.rnn(rnn_input)  # (batch x seq_len x hidden_dim)
        
        # Predict parameters for each position
        params = self.param_head(rnn_output)  # (batch x seq_len x 10)
        
        # Apply sigmoid to normalize to [0, 1] range
        params = torch.sigmoid(params)
        
        return params


class EnhancedPVCv2Model(nn.Module):
    """
    Enhanced PVC v2.0 model with full sequence parameter prediction.
    
    Improvements over original:
    - Predicts parameters for ALL functions in sequence
    - Enables meaningful reconstruction and PSNR measurement
    - Combined loss: function IDs + parameters
    """
    
    def __init__(self,
                 feature_dim: int = 256,
                 hidden_dim: int = 128,
                 num_functions: int = 42,
                 max_sequence_length: int = 20):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_functions = num_functions
        self.max_sequence_length = max_sequence_length
        
        # Components
        self.encoder = CNNEncoder(feature_dim)
        self.decoder = RNNDecoder(feature_dim, hidden_dim, num_functions, max_sequence_length)
        self.param_predictor = SequenceParameterPredictor(feature_dim, hidden_dim, num_functions)
    
    def forward(self,
                frames: torch.Tensor,
                teacher_forcing_sequences: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with full parameter prediction.
        
        Args:
            frames: Input frames (batch_size x 3 x H x W)
            teacher_forcing_sequences: Ground truth function IDs (batch_size x seq_len)
            
        Returns:
            function_logits: (batch_size x seq_len x num_functions+1)
            predicted_sequences: (batch_size x seq_len) - predicted function IDs
            predicted_params: (batch_size x seq_len x 10) - predicted parameters
        """
        # Encode frames
        features = self.encoder(frames)  # (batch_size x feature_dim)
        
        # Decode to function sequence
        function_logits, predicted_sequences = self.decoder(features, teacher_forcing_sequences)
        # function_logits: (batch x seq_len x num_functions+1)
        # predicted_sequences: (batch x seq_len)
        
        # Predict parameters for the sequence
        # Use teacher forcing sequences if provided, otherwise use predictions
        if teacher_forcing_sequences is not None:
            func_ids_for_params = teacher_forcing_sequences
        else:
            func_ids_for_params = predicted_sequences
        
        predicted_params = self.param_predictor(features, func_ids_for_params)
        # predicted_params: (batch x seq_len x 10)
        
        return function_logits, predicted_sequences, predicted_params
    
    def predict_with_params(self, frame):
        """
        Predict function sequence and parameters for a single frame.
        
        Args:
            frame: Input frame (H x W x 3), uint8
            
        Returns:
            function_ids: List of function IDs
            parameters: List of parameter arrays (10 each)
        """
        import numpy as np
        
        self.eval()
        with torch.no_grad():
            # Prepare input
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            device = next(self.parameters()).device
            frame_tensor = frame_tensor.to(device)
            
            # Forward pass
            _, predicted_sequences, predicted_params = self(frame_tensor)
            
            # Convert to lists
            func_ids = predicted_sequences[0].cpu().numpy().tolist()
            params = predicted_params[0].cpu().numpy()  # (seq_len x 10)
            
            # Filter out END tokens
            valid_funcs = []
            valid_params = []
            
            for fid, param in zip(func_ids, params):
                if fid >= self.num_functions:  # END token
                    break
                valid_funcs.append(int(fid))
                valid_params.append(param)
            
            return valid_funcs, valid_params


if __name__ == "__main__":
    # Quick test
    print("Testing EnhancedPVCv2Model...")
    
    model = EnhancedPVCv2Model(
        feature_dim=256,
        hidden_dim=128,
        num_functions=42,
        max_sequence_length=20
    )
    
    # Test forward pass
    batch_size = 4
    frames = torch.randn(batch_size, 3, 256, 256)
    teacher_forcing = torch.randint(0, 42, (batch_size, 10))
    
    func_logits, pred_seqs, pred_params = model(frames, teacher_forcing)
    
    print(f"✅ Forward pass successful!")
    print(f"   Function logits shape: {func_logits.shape}")
    print(f"   Predicted sequences shape: {pred_seqs.shape}")
    print(f"   Predicted params shape: {pred_params.shape}")
    print(f"\n   Expected: (batch={batch_size}, seq_len=10)")
    print(f"   Function logits: ({batch_size}, 10, 43)")  # 42 funcs + 1 END
    print(f"   Sequences: ({batch_size}, 10)")
    print(f"   Params: ({batch_size}, 10, 10)")  # 10 params per function
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n   Total parameters: {total_params:,}")
    
    print("\n✅ Model ready for training!")

