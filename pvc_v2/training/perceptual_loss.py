"""
Perceptual Loss Module for PVC v2.0

Uses VGG16 features to compute perceptual similarity.
This helps the model focus on visual quality rather than just pixel-level accuracy.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss based on VGG16 features.
    
    Compares high-level features from pre-trained VGG rather than raw pixels.
    This correlates better with human perception.
    """
    
    def __init__(self, feature_layers=[3, 8, 15, 22], device='cpu'):
        """
        Args:
            feature_layers: Which VGG conv layers to use for feature extraction
                           Default: conv1_2, conv2_2, conv3_3, conv4_3
            device: torch device
        """
        super(VGGPerceptualLoss, self).__init__()
        
        # Load pre-trained VGG16
        vgg = models.vgg16(pretrained=True).features.to(device).eval()
        
        # Freeze all parameters (we're only using for feature extraction)
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Split VGG into blocks at specified layers
        self.blocks = nn.ModuleList()
        prev_layer = 0
        
        for layer_idx in feature_layers:
            block = nn.Sequential(*[vgg[i] for i in range(prev_layer, layer_idx + 1)])
            self.blocks.append(block)
            prev_layer = layer_idx + 1
        
        # VGG normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.device = device
    
    def normalize(self, img):
        """
        Normalize image for VGG (expects ImageNet normalization).
        
        Args:
            img: Tensor (B, 3, H, W) in [0, 1] range
            
        Returns:
            Normalized tensor
        """
        return (img - self.mean.to(img.device)) / self.std.to(img.device)
    
    def extract_features(self, x):
        """
        Extract features from multiple VGG layers.
        
        Args:
            x: Input tensor (B, 3, H, W) in [0, 1] range
            
        Returns:
            List of feature tensors from each block
        """
        x = self.normalize(x)
        features = []
        
        for block in self.blocks:
            x = block(x)
            features.append(x)
        
        return features
    
    def forward(self, pred, target):
        """
        Compute perceptual loss between prediction and target.
        
        Args:
            pred: Predicted image (B, 3, H, W) in [0, 1] range
            target: Target image (B, 3, H, W) in [0, 1] range
            
        Returns:
            Perceptual loss (scalar)
        """
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += F.mse_loss(pred_feat, target_feat)
        
        # Average across all feature layers
        loss = loss / len(pred_features)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for PVC v2.0 training:
    - Function prediction loss (CrossEntropy)
    - Parameter prediction loss (MSE)
    - Perceptual loss (VGG-based)
    """
    
    def __init__(self, 
                 num_functions,
                 device='cpu',
                 weight_function=0.3,
                 weight_param=0.3,
                 weight_perceptual=0.4):
        """
        Args:
            num_functions: Number of function classes
            device: torch device
            weight_function: Weight for function prediction loss
            weight_param: Weight for parameter prediction loss
            weight_perceptual: Weight for perceptual loss
        """
        super(CombinedLoss, self).__init__()
        
        self.function_criterion = nn.CrossEntropyLoss()
        self.param_criterion = nn.MSELoss()
        self.perceptual_loss = VGGPerceptualLoss(device=device)
        
        self.weight_function = weight_function
        self.weight_param = weight_param
        self.weight_perceptual = weight_perceptual
        
        self.num_functions = num_functions
        self.device = device
    
    def forward(self, 
                function_logits, 
                predicted_params,
                predicted_sequences,
                target_func_ids,
                target_params,
                reconstructed_frames,
                original_frames):
        """
        Compute combined loss.
        
        Args:
            function_logits: (B, seq_len, num_functions)
            predicted_params: (B, seq_len, 10)
            predicted_sequences: (B, seq_len) - predicted function IDs
            target_func_ids: (B, seq_len)
            target_params: (B, seq_len, 10)
            reconstructed_frames: (B, 3, H, W) - reconstructed images in [0, 1]
            original_frames: (B, 3, H, W) - original images in [0, 1]
            
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary of individual losses for logging
        """
        batch_size = function_logits.size(0)
        seq_len = min(function_logits.size(1), target_func_ids.size(1))
        
        # 1. Function prediction loss
        func_loss = 0
        for t in range(seq_len):
            func_loss += self.function_criterion(
                function_logits[:, t, :],
                target_func_ids[:, t]
            )
        func_loss = func_loss / seq_len
        
        # 2. Parameter prediction loss (only for non-END tokens)
        mask = (target_func_ids[:, :seq_len] != self.num_functions).float()
        if mask.sum() > 0:
            param_loss = self.param_criterion(
                predicted_params[:, :seq_len] * mask.unsqueeze(-1),
                target_params[:, :seq_len] * mask.unsqueeze(-1)
            )
        else:
            param_loss = torch.tensor(0.0, device=self.device)
        
        # 3. Perceptual loss (only if reconstructed frames provided)
        if reconstructed_frames is not None and original_frames is not None:
            perceptual_loss = self.perceptual_loss(reconstructed_frames, original_frames)
        else:
            perceptual_loss = torch.tensor(0.0, device=self.device)
        
        # Combine losses
        total_loss = (
            self.weight_function * func_loss +
            self.weight_param * param_loss +
            self.weight_perceptual * perceptual_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'function': func_loss.item(),
            'param': param_loss.item(),
            'perceptual': perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss
        }
        
        return total_loss, loss_dict


def test_perceptual_loss():
    """Quick test of perceptual loss module."""
    print("Testing VGG Perceptual Loss...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create loss module
    perc_loss = VGGPerceptualLoss(device=device)
    
    # Test with random images
    img1 = torch.rand(2, 3, 256, 256).to(device)
    img2 = torch.rand(2, 3, 256, 256).to(device)
    
    loss = perc_loss(img1, img2)
    print(f"Random images loss: {loss.item():.4f}")
    
    # Test with identical images
    loss_identical = perc_loss(img1, img1)
    print(f"Identical images loss: {loss_identical.item():.6f}")
    
    # Test with slightly different images
    img3 = img1 + torch.randn_like(img1) * 0.05
    loss_similar = perc_loss(img1, img3)
    print(f"Similar images loss: {loss_similar.item():.4f}")
    
    print("\n✅ Perceptual loss working correctly!")
    print(f"   Identical: {loss_identical.item():.6f}")
    print(f"   Similar: {loss_similar.item():.4f}")
    print(f"   Different: {loss.item():.4f}")


if __name__ == "__main__":
    test_perceptual_loss()

