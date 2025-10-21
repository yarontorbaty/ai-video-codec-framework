#!/usr/bin/env python3
"""
Neural Network Architecture for PVC v2.0

CNN Encoder: Frame → Visual features
RNN Decoder: Features → Function sequence
Parameter Predictor: Function → Parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np


class CNNEncoder(nn.Module):
    """
    CNN Encoder: Extracts visual features from frames.
    
    Input: RGB frame (3 x H x W)
    Output: Feature vector (feature_dim)
    """
    
    def __init__(self, feature_dim: int = 256):
        """
        Initialize CNN encoder.
        
        Args:
            feature_dim: Size of output feature vector
        """
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # → 32 x H/2 x W/2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # → 64 x H/4 x W/4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # → 128 x H/8 x W/8
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # → 256 x H/16 x W/16
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Global average pooling + FC
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input frame (batch_size x 3 x H x W)
            
        Returns:
            Feature vector (batch_size x feature_dim)
        """
        # Convolutional blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # FC layer
        x = self.fc(x)
        
        return x


class RNNDecoder(nn.Module):
    """
    RNN Decoder: Generates sequence of function IDs.
    
    Input: Feature vector from encoder
    Output: Sequence of function IDs
    """
    
    def __init__(self, 
                 feature_dim: int = 256,
                 hidden_dim: int = 256,
                 num_functions: int = 10,
                 max_sequence_length: int = 20):
        """
        Initialize RNN decoder.
        
        Args:
            feature_dim: Size of input feature vector
            hidden_dim: Size of LSTM hidden state
            num_functions: Number of possible function types
            max_sequence_length: Maximum sequence length
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_functions = num_functions
        self.max_sequence_length = max_sequence_length
        
        # LSTM for sequence generation
        self.lstm = nn.LSTM(feature_dim + num_functions, hidden_dim, batch_first=True)
        
        # Output layer: predict next function ID
        self.fc_out = nn.Linear(hidden_dim, num_functions + 1)  # +1 for END token
        
        # Embedding for function IDs
        self.func_embedding = nn.Embedding(num_functions + 1, num_functions)
    
    def forward(self, 
                features: torch.Tensor,
                teacher_forcing_sequences: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Visual features from encoder (batch_size x feature_dim)
            teacher_forcing_sequences: Ground truth sequences for training (batch_size x seq_len)
            
        Returns:
            (function_logits, predicted_sequences)
            function_logits: (batch_size x max_seq_len x num_functions+1)
            predicted_sequences: (batch_size x max_seq_len)
        """
        batch_size = features.size(0)
        device = features.device
        
        # Initialize hidden state
        h = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        
        # Start token (all zeros)
        current_input = torch.zeros(batch_size, self.num_functions).to(device)
        
        # Collect outputs
        all_logits = []
        all_predictions = []
        
        for t in range(self.max_sequence_length):
            # Concatenate features with current input
            lstm_input = torch.cat([features, current_input], dim=1).unsqueeze(1)
            
            # LSTM step
            output, (h, c) = self.lstm(lstm_input, (h, c))
            
            # Predict next function
            logits = self.fc_out(output.squeeze(1))
            all_logits.append(logits)
            
            # Get prediction
            prediction = torch.argmax(logits, dim=1)
            all_predictions.append(prediction)
            
            # Prepare next input
            if teacher_forcing_sequences is not None and t < teacher_forcing_sequences.size(1):
                # Teacher forcing: use ground truth
                next_func = teacher_forcing_sequences[:, t]
            else:
                # Use prediction
                next_func = prediction
            
            # Embed function ID
            current_input = self.func_embedding(next_func)
            
            # Note: Don't stop early during training with teacher forcing
            # Only stop when doing inference without teacher forcing
            if teacher_forcing_sequences is None and (prediction == self.num_functions).all():
                break
        
        # Stack outputs
        function_logits = torch.stack(all_logits, dim=1)  # (batch x seq_len x num_functions+1)
        predicted_sequences = torch.stack(all_predictions, dim=1)  # (batch x seq_len)
        
        return function_logits, predicted_sequences


class ParameterPredictor(nn.Module):
    """
    Parameter Predictor: Predicts parameters for each function.
    
    Input: Visual features + function ID
    Output: Function parameters
    """
    
    def __init__(self, feature_dim: int = 256, num_functions: int = 10):
        """
        Initialize parameter predictor.
        
        Args:
            feature_dim: Size of visual feature vector
            num_functions: Number of possible function types
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_functions = num_functions
        
        # Function embedding (+1 for END token)
        self.func_embedding = nn.Embedding(num_functions + 1, 64)
        
        # Parameter prediction network
        self.fc1 = nn.Linear(feature_dim + 64, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Different heads for different parameter types
        self.coord_head = nn.Linear(256, 4)  # x, y, w, h (or cx, cy, rx, ry)
        self.color1_head = nn.Linear(256, 3)  # RGB (0-1)
        self.color2_head = nn.Linear(256, 3)  # RGB (0-1) for gradients
        self.scalar_head = nn.Linear(256, 4)  # opacity, angle, radius, stroke_width
    
    def forward(self, features: torch.Tensor, func_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Visual features (batch_size x feature_dim)
            func_ids: Function IDs (batch_size)
            
        Returns:
            Dictionary of predicted parameters
        """
        # Embed function ID
        func_emb = self.func_embedding(func_ids)
        
        # Concatenate features and function embedding
        x = torch.cat([features, func_emb], dim=1)
        
        # Shared layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Predict parameters
        coords = self.coord_head(x)  # (x, y, w, h) or (cx, cy, rx, ry)
        color1 = torch.sigmoid(self.color1_head(x))  # RGB in [0, 1]
        color2 = torch.sigmoid(self.color2_head(x))  # RGB in [0, 1]
        scalars = self.scalar_head(x)  # Various scalars
        
        return {
            'coords': coords,
            'color1': color1,
            'color2': color2,
            'scalars': scalars
        }


class PVCv2Model(nn.Module):
    """
    Complete PVC v2.0 model: Frame → Function sequence + parameters.
    """
    
    def __init__(self,
                 feature_dim: int = 256,
                 hidden_dim: int = 256,
                 num_functions: int = 10,
                 max_sequence_length: int = 20):
        """
        Initialize PVC v2.0 model.
        
        Args:
            feature_dim: Size of feature vectors
            hidden_dim: Size of LSTM hidden state
            num_functions: Number of possible function types
            max_sequence_length: Maximum sequence length
        """
        super().__init__()
        
        self.encoder = CNNEncoder(feature_dim)
        self.decoder = RNNDecoder(feature_dim, hidden_dim, num_functions, max_sequence_length)
        self.param_predictor = ParameterPredictor(feature_dim, num_functions)
        
        self.feature_dim = feature_dim
        self.num_functions = num_functions
    
    def forward(self, 
                frames: torch.Tensor,
                teacher_forcing_sequences: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            frames: Input frames (batch_size x 3 x H x W)
            teacher_forcing_sequences: Ground truth sequences (batch_size x seq_len) for training
            
        Returns:
            (function_logits, predicted_sequences, parameters)
        """
        # Encode frames
        features = self.encoder(frames)
        
        # Decode to function sequence
        function_logits, predicted_sequences = self.decoder(features, teacher_forcing_sequences)
        
        # Predict parameters for first function (for demonstration)
        # In practice, you'd predict params for all functions in the sequence
        first_func = predicted_sequences[:, 0]
        parameters = self.param_predictor(features, first_func)
        
        return function_logits, predicted_sequences, parameters
    
    def predict_sequence(self, frame: np.ndarray) -> List[Dict]:
        """
        Predict complete function sequence for a frame.
        
        Args:
            frame: Input frame (H x W x 3), uint8, 0-255
            
        Returns:
            List of function calls with parameters
        """
        self.eval()
        with torch.no_grad():
            # Prepare input
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            device = next(self.parameters()).device
            frame_tensor = frame_tensor.to(device)
            
            # Encode
            features = self.encoder(frame_tensor)
            
            # Decode sequence
            _, predicted_sequence = self.decoder(features)
            
            # Predict parameters for each function
            function_calls = []
            for func_id in predicted_sequence[0]:
                func_id_val = func_id.item()
                
                # Stop at END token
                if func_id_val >= self.num_functions:
                    break
                
                # Predict parameters
                params = self.param_predictor(features, func_id.unsqueeze(0))
                
                # Convert to dict
                function_calls.append({
                    'func_id': func_id_val,
                    'coords': params['coords'][0].cpu().numpy(),
                    'color1': params['color1'][0].cpu().numpy(),
                    'color2': params['color2'][0].cpu().numpy(),
                    'scalars': params['scalars'][0].cpu().numpy(),
                })
            
            return function_calls


if __name__ == '__main__':
    """Test the model architecture."""
    print("Testing PVC v2.0 Neural Network...")
    
    # Create model
    model = PVCv2Model(
        feature_dim=256,
        hidden_dim=256,
        num_functions=10,
        max_sequence_length=15
    )
    
    # Test with dummy data
    batch_size = 4
    frames = torch.randn(batch_size, 3, 256, 256)
    teacher_sequences = torch.randint(0, 10, (batch_size, 10))
    
    # Forward pass
    function_logits, predicted_sequences, parameters = model(frames, teacher_sequences)
    
    print(f"\n✅ Model architecture test passed!")
    print(f"   Input: {frames.shape}")
    print(f"   Function logits: {function_logits.shape}")
    print(f"   Predicted sequences: {predicted_sequences.shape}")
    print(f"   Coords: {parameters['coords'].shape}")
    print(f"   Color1: {parameters['color1'].shape}")
    print(f"   Color2: {parameters['color2'].shape}")
    print(f"   Scalars: {parameters['scalars'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model size: {total_params:,} parameters ({total_params * 4 / 1024 / 1024:.2f} MB)")

