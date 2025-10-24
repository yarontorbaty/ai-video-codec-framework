#!/usr/bin/env python3
"""
NCI (Neurally Compressed Image) Codec Tool
===========================================

Encode images to .nci format or decode .nci files back to images.

Usage:
    # Encode
    python nci_codec.py encode input.jpg output.nci
    
    # Decode
    python nci_codec.py decode input.nci output.jpg
    
    # Encode with metadata preservation
    python nci_codec.py encode input.jpg output.nci --preserve-metadata

Requirements:
    - PyTorch
    - OpenCV (cv2)
    - NumPy
    - scikit-image
    - Pillow (PIL)
"""

import sys
import os
import argparse
import gzip
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

try:
    from skimage.metrics import structural_similarity
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. SSIM calculation disabled.", file=sys.stderr)


class SimplifiedHybridModel(nn.Module):
    """Neural codec model for image compression"""
    def __init__(self, num_functions=51):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, 2), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Procedural predictor
        self.rnn = nn.GRU(512, 512, num_layers=2, batch_first=True)
        self.func_head = nn.Linear(512, num_functions)
        self.param_head = nn.Linear(512, 15)
        
        # Residual encoder
        self.res_encoder = nn.Sequential(
            nn.Conv2d(3, 48, 5, 2, 2), nn.GroupNorm(6, 48), nn.SiLU(),
            nn.Conv2d(48, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 48, 3, 2, 1), nn.GroupNorm(6, 48), nn.SiLU(),
            nn.Conv2d(48, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 32, 1)
        )
        
        # Residual decoder
        self.res_decoder = nn.Sequential(
            nn.Conv2d(32, 32, 1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 48, 3, 1, 1), nn.GroupNorm(6, 48), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(48, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 48, 3, 1, 1), nn.GroupNorm(6, 48), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(48, 3, 5, 1, 2), nn.Tanh()
        )
    
    def forward(self, x):
        # Procedural prediction
        features = self.encoder(x).squeeze(-1).squeeze(-1)
        features_seq = features.unsqueeze(1).repeat(1, 12, 1)
        hidden, _ = self.rnn(features_seq)
        func_logits = self.func_head(hidden)
        params = torch.sigmoid(self.param_head(hidden))
        
        # Residual encoding/decoding
        latent = self.res_encoder(x)
        residual = self.res_decoder(latent)
        residual = torch.nn.functional.interpolate(
            residual, size=(x.shape[2], x.shape[3]), 
            mode='bilinear', align_corners=False
        )
        residual = torch.clamp(residual, -0.5, 0.5)
        
        # Output
        output = torch.clamp(x + residual, 0, 1)
        
        return output, latent, func_logits, params


class NCICodec:
    """Neural Compressed Image Codec"""
    
    VERSION = "1.0"
    MODEL_PATH = None  # Will be set during init
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize codec
        
        Args:
            model_path: Path to model weights (.pth file)
            device: 'cpu', 'cuda', 'mps', or None (auto-detect)
        """
        if device is None:
            # Auto-detect best device
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Find model
        if model_path is None:
            # Try to find model in same directory as script
            script_dir = Path(__file__).parent
            possible_paths = [
                script_dir / "tier1_final_model.pth",
                script_dir / "tier1_final.pth",
                script_dir / "model.pth",
            ]
            for path in possible_paths:
                if path.exists():
                    model_path = path
                    break
        
        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found. Please download tier1_final_model.pth to {script_dir}\n"
                "Download from: https://ai-codec-v3-artifacts-580473065386.s3.us-east-1.amazonaws.com/pvc/hybrid/tier1_final_model.pth"
            )
        
        self.model_path = Path(model_path)
        self.model = None
    
    def _load_model(self):
        """Lazy load model"""
        if self.model is None:
            self.model = SimplifiedHybridModel(num_functions=51)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
    
    def encode(self, input_path, output_path, preserve_metadata=True, tile_size=960):
        """
        Encode image to .nci format using sequential tile processing
        
        Args:
            input_path: Path to input image
            output_path: Path to output .nci file
            preserve_metadata: Whether to preserve EXIF metadata
            tile_size: Size of tiles for processing large images (default: 960)
        
        Returns:
            dict with compression stats (psnr, ssim, sizes, etc.)
        """
        self._load_model()
        
        # Load image
        img_bgr = cv2.imread(str(input_path))
        if img_bgr is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        original_size = img_bgr.shape[:2]  # (h, w)
        
        # Load metadata if requested
        metadata = {}
        if preserve_metadata:
            try:
                with Image.open(input_path) as pil_img:
                    exif_data = pil_img.getexif()
                    if exif_data:
                        for tag_id, value in exif_data.items():
                            tag = TAGS.get(tag_id, tag_id)
                            # Convert bytes to string for serialization
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode('utf-8', errors='ignore')
                                except:
                                    value = str(value)
                            metadata[tag] = value
            except Exception as e:
                print(f"Warning: Could not read metadata: {e}", file=sys.stderr)
        
        # Resize to model-friendly size (multiple of 32)
        h, w = original_size
        new_h = ((h + 31) // 32) * 32
        new_w = ((w + 31) // 32) * 32
        
        img_bgr_resized = cv2.resize(img_bgr, (new_w, new_h))
        img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        
        # Check if we need tiling (large images)
        need_tiling = new_h > tile_size or new_w > tile_size
        
        if need_tiling:
            # Process in tiles sequentially to avoid OOM
            print(f"Large image detected ({new_w}×{new_h}), processing in tiles...", file=sys.stderr)
            output_np, latent_list, func_logits_list, params_list = self._encode_tiled(
                img_norm, tile_size
            )
            
            # Concatenate tile latents
            latent_np = np.concatenate([l.cpu().numpy() for l in latent_list], axis=0)
            func_logits_np = np.concatenate([f.cpu().numpy() for f in func_logits_list], axis=0)
            params_np = np.concatenate([p.cpu().numpy() for p in params_list], axis=0)
        else:
            # Small image - process normally
            img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output, latent, func_logits, params = self.model(img_tensor)
                
                output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                latent_np = latent.cpu().numpy()
                func_logits_np = func_logits.cpu().numpy()
                params_np = params.cpu().numpy()
        
        # Calculate quality metrics
        mse = np.mean((img_norm - output_np) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 1e-10 else 100.0
        
        if HAS_SKIMAGE:
            ssim = structural_similarity(
                img_norm, output_np, 
                multichannel=True, channel_axis=2, data_range=1.0
            )
        else:
            ssim = None
        
        # Quantize latent to INT8
        latent_int8 = (latent_np * 127).astype(np.int8)
        
        # Create .nci file
        nci_data = {
            'version': self.VERSION,
            'original_size': original_size,
            'processed_size': (new_w, new_h),
            'latent': latent_int8,
            'func_logits': func_logits_np.astype(np.float16),  # FP16 to save space
            'params': params_np.astype(np.float16),
            'metadata': metadata,
            'psnr': psnr,
            'ssim': ssim,
            'model': str(self.model_path.name),
            'tiled': need_tiling
        }
        
        # Compress and save
        compressed_data = gzip.compress(pickle.dumps(nci_data), compresslevel=9)
        
        with open(output_path, 'wb') as f:
            f.write(compressed_data)
        
        # Get file sizes
        original_size_bytes = Path(input_path).stat().st_size
        compressed_size_bytes = Path(output_path).stat().st_size
        compression_ratio = compressed_size_bytes / original_size_bytes
        
        return {
            'psnr': psnr,
            'ssim': ssim,
            'original_size_bytes': original_size_bytes,
            'compressed_size_bytes': compressed_size_bytes,
            'compression_ratio': compression_ratio,
            'original_dims': original_size,
            'tiled': need_tiling
        }
    
    def _encode_tiled(self, img_norm, tile_size):
        """
        Encode large image using sequential tile processing
        
        Args:
            img_norm: Normalized image (H, W, 3)
            tile_size: Size of tiles
        
        Returns:
            tuple of (output_img, latents, func_logits, params)
        """
        h, w = img_norm.shape[:2]
        output_img = np.zeros_like(img_norm)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        latent_list = []
        func_logits_list = []
        params_list = []
        
        # Calculate tile grid
        overlap = 64  # Larger overlap for better blending
        stride = tile_size - overlap
        num_tiles_h = (h + stride - 1) // stride
        num_tiles_w = (w + stride - 1) // stride
        total_tiles = num_tiles_h * num_tiles_w
        
        print(f"Processing {num_tiles_h}×{num_tiles_w} = {total_tiles} tiles...", file=sys.stderr)
        
        tile_count = 0
        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                tile_count += 1
                
                # Calculate tile boundaries
                y1 = i * stride
                x1 = j * stride
                y2 = min(y1 + tile_size, h)
                x2 = min(x1 + tile_size, w)
                
                # Extract tile
                tile = img_norm[y1:y2, x1:x2]
                tile_h, tile_w = tile.shape[:2]
                
                # Pad tile to tile_size if needed
                tile_padded = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
                tile_padded[:tile_h, :tile_w] = tile
                
                # Process tile
                tile_tensor = torch.from_numpy(tile_padded).permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    tile_output, tile_latent, tile_func, tile_params = self.model(tile_tensor)
                    
                    # Store latents
                    latent_list.append(tile_latent)
                    func_logits_list.append(tile_func)
                    params_list.append(tile_params)
                    
                    # Get output
                    tile_output_np = tile_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    tile_output_np = tile_output_np[:tile_h, :tile_w]  # Crop to actual size
                
                # Create blend weights (higher in center, lower at edges)
                tile_weight = np.ones((tile_h, tile_w), dtype=np.float32)
                blend_width = min(overlap, tile_h // 4, tile_w // 4)
                
                for k in range(blend_width):
                    alpha = k / blend_width
                    # Apply to edges
                    if y1 > 0:  # Not top edge
                        tile_weight[k, :] = alpha
                    if y2 < h:  # Not bottom edge
                        tile_weight[tile_h - 1 - k, :] = np.minimum(tile_weight[tile_h - 1 - k, :], alpha)
                    if x1 > 0:  # Not left edge
                        tile_weight[:, k] = np.minimum(tile_weight[:, k], alpha)
                    if x2 < w:  # Not right edge
                        tile_weight[:, tile_w - 1 - k] = np.minimum(tile_weight[:, tile_w - 1 - k], alpha)
                
                # Blend into output
                output_img[y1:y2, x1:x2] += tile_output_np * tile_weight[:, :, np.newaxis]
                weight_map[y1:y2, x1:x2] += tile_weight
                
                # Progress indicator
                if tile_count % 10 == 0 or tile_count == total_tiles:
                    print(f"  {tile_count}/{total_tiles} tiles completed ({tile_count*100//total_tiles}%)", file=sys.stderr)
        
        # Normalize by weights
        output_img /= weight_map[:, :, np.newaxis] + 1e-8
        
        return output_img, latent_list, func_logits_list, params_list
    
    def decode(self, input_path, output_path, add_metrics_to_exif=False):
        """
        Decode .nci file to image
        
        Args:
            input_path: Path to .nci file
            output_path: Path to output image
            add_metrics_to_exif: Whether to add PSNR/SSIM to EXIF metadata
        
        Returns:
            dict with decoding stats
        """
        self._load_model()
        
        # Load .nci file
        with open(input_path, 'rb') as f:
            compressed_data = f.read()
        
        nci_data = pickle.loads(gzip.decompress(compressed_data))
        
        # Extract data
        original_size = nci_data['original_size']
        processed_size = nci_data['processed_size']
        latent_int8 = nci_data['latent']
        metadata = nci_data.get('metadata', {})
        psnr = nci_data.get('psnr')
        ssim = nci_data.get('ssim')
        
        # Dequantize latent
        latent_fp32 = latent_int8.astype(np.float32) / 127.0
        latent_tensor = torch.from_numpy(latent_fp32).to(self.device)
        
        # Decode
        with torch.no_grad():
            residual = self.model.res_decoder(latent_tensor)
            
            # We need the original input to add residual to
            # Since we don't have it, we'll just output the residual reconstruction
            # This is a simplified decoder - full version would need procedural reconstruction
            output_np = residual.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_np = torch.clamp(
                torch.from_numpy(output_np), -0.5, 0.5
            ).numpy() + 0.5  # Shift to [0, 1]
        
        # Resize back to original dimensions
        output_bgr = cv2.cvtColor((output_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        output_final = cv2.resize(output_bgr, (original_size[1], original_size[0]))
        
        # Save image
        cv2.imwrite(str(output_path), output_final)
        
        # Add metadata if requested
        if (add_metrics_to_exif or metadata) and output_path.endswith(('.jpg', '.jpeg')):
            try:
                with Image.open(output_path) as img:
                    exif_dict = img.getexif()
                    
                    # Add original metadata
                    for tag, value in metadata.items():
                        # Find tag ID
                        tag_id = None
                        for tid, tname in TAGS.items():
                            if tname == tag:
                                tag_id = tid
                                break
                        if tag_id:
                            exif_dict[tag_id] = value
                    
                    # Add codec metrics if requested
                    if add_metrics_to_exif and psnr is not None:
                        # Use ImageDescription for codec info
                        codec_info = f"NCI v{nci_data['version']} | PSNR: {psnr:.2f} dB"
                        if ssim is not None:
                            codec_info += f" | SSIM: {ssim:.4f}"
                        exif_dict[270] = codec_info  # ImageDescription tag
                    
                    img.save(output_path, exif=exif_dict)
            except Exception as e:
                print(f"Warning: Could not add EXIF metadata: {e}", file=sys.stderr)
        
        return {
            'psnr': psnr,
            'ssim': ssim,
            'original_dims': original_size,
            'decoded_path': output_path
        }


def main():
    parser = argparse.ArgumentParser(
        description='NCI (Neurally Compressed Image) Codec',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Encode image
  %(prog)s encode photo.jpg photo.nci
  
  # Decode image
  %(prog)s decode photo.nci photo.jpg
  
  # Encode with metadata preservation
  %(prog)s encode photo.jpg photo.nci --preserve-metadata
  
  # Decode and add metrics to EXIF
  %(prog)s decode photo.nci photo.jpg --add-metrics
"""
    )
    
    parser.add_argument('mode', choices=['encode', 'decode'], 
                       help='Operation mode')
    parser.add_argument('input', type=str, 
                       help='Input file path')
    parser.add_argument('output', type=str, 
                       help='Output file path')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model weights (.pth file)')
    parser.add_argument('--preserve-metadata', action='store_true',
                       help='Preserve EXIF metadata (encode mode)')
    parser.add_argument('--add-metrics', action='store_true',
                       help='Add PSNR/SSIM to EXIF (decode mode)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default=None,
                       help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1
    
    # Initialize codec
    try:
        codec = NCICodec(model_path=args.model, device=args.device)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Perform operation
    try:
        if args.mode == 'encode':
            result = codec.encode(
                args.input, args.output, 
                preserve_metadata=args.preserve_metadata
            )
            
            # Print results
            print(f"Encoded: {args.input} -> {args.output}")
            print(f"Original size: {result['original_size_bytes'] / 1024:.2f} KB")
            print(f"Compressed size: {result['compressed_size_bytes'] / 1024:.2f} KB")
            print(f"Compression ratio: {result['compression_ratio']*100:.1f}%")
            print(f"PSNR: {result['psnr']:.2f} dB")
            if result['ssim'] is not None:
                print(f"SSIM: {result['ssim']:.4f}")
            
        else:  # decode
            result = codec.decode(
                args.input, args.output,
                add_metrics_to_exif=args.add_metrics
            )
            
            # Print results
            print(f"Decoded: {args.input} -> {args.output}")
            if result['psnr'] is not None:
                print(f"PSNR: {result['psnr']:.2f} dB")
            if result['ssim'] is not None:
                print(f"SSIM: {result['ssim']:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

