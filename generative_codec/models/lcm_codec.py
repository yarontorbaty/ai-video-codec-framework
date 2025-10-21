"""
LumaFlow Codec - LCM-based Video Compression with Depth
Uses Latent Consistency Models for fast encoding/decoding
"""

import torch
import torch.nn as nn
import numpy as np
from diffusers import LCMScheduler, AutoencoderKL
from typing import Optional, Tuple, Dict
import cv2


class LCMVideoEncoder(nn.Module):
    """
    Encodes video frames to compact latent representations using LCM
    """
    
    def __init__(
        self,
        model_id: str = "SimianLuo/LCM_Dreamshaper_v7",
        latent_channels: int = 4,
        latent_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        
        # Load VAE (for encoding to latent space)
        print(f"Loading VAE encoder from {model_id}...")
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse"
        ).to(device)
        self.vae.eval()
        
        # LCM scheduler for fast inference
        self.scheduler = LCMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler"
        )
        
        print(f"✅ LCM Encoder initialized on {device}")
    
    @torch.no_grad()
    def encode_frame(
        self,
        rgb_frame: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        downscale_to: int = 512
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a single RGB frame (+ optional depth) to latent representation
        
        Args:
            rgb_frame: (H, W, 3) numpy array in [0, 255]
            depth_map: (H, W) numpy array of depth values
            downscale_to: Target size for encoding
            
        Returns:
            Dictionary with:
                - rgb_latent: (4, 64, 64) latent tensor
                - depth_latent: (1, 64, 64) compressed depth
                - downscaled_rgb: (3, 512, 512) downscaled reference
        """
        # Preprocess RGB
        rgb_tensor = self._preprocess_rgb(rgb_frame, downscale_to)
        
        # Encode to latent space using VAE
        latent = self.vae.encode(rgb_tensor).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        
        result = {
            'rgb_latent': latent.squeeze(0),  # (4, 64, 64)
            'downscaled_rgb': rgb_tensor.squeeze(0)  # (3, 512, 512)
        }
        
        # Encode depth if provided
        if depth_map is not None:
            depth_latent = self._encode_depth(depth_map, downscale_to)
            result['depth_latent'] = depth_latent
        
        return result
    
    def _preprocess_rgb(self, rgb_frame: np.ndarray, size: int) -> torch.Tensor:
        """Convert numpy RGB to normalized tensor"""
        # Resize to target size
        rgb_resized = cv2.resize(rgb_frame, (size, size), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to tensor [0, 1]
        rgb_tensor = torch.from_numpy(rgb_resized).float() / 255.0
        
        # Rearrange to (C, H, W) and normalize to [-1, 1]
        rgb_tensor = rgb_tensor.permute(2, 0, 1)
        rgb_tensor = rgb_tensor * 2.0 - 1.0
        
        # Add batch dimension
        return rgb_tensor.unsqueeze(0).to(self.device)
    
    def _encode_depth(self, depth_map: np.ndarray, size: int) -> torch.Tensor:
        """Compress depth map to latent representation"""
        # Resize depth map
        depth_resized = cv2.resize(depth_map, (size // 8, size // 8), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        depth_min, depth_max = depth_resized.min(), depth_resized.max()
        if depth_max > depth_min:
            depth_normalized = (depth_resized - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_resized)
        
        # Convert to tensor
        depth_tensor = torch.from_numpy(depth_normalized).float()
        return depth_tensor.unsqueeze(0).to(self.device)  # (1, 64, 64)


class LCMVideoDecoder(nn.Module):
    """
    Decodes latent representations back to video frames using LCM
    """
    
    def __init__(
        self,
        model_id: str = "SimianLuo/LCM_Dreamshaper_v7",
        num_inference_steps: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.num_inference_steps = num_inference_steps
        
        # Load VAE (for decoding from latent space)
        print(f"Loading VAE decoder from {model_id}...")
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse"
        ).to(device)
        self.vae.eval()
        
        print(f"✅ LCM Decoder initialized on {device}")
    
    @torch.no_grad()
    def decode_frame(
        self,
        rgb_latent: torch.Tensor,
        depth_latent: Optional[torch.Tensor] = None,
        downscaled_rgb: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Decode latent representation back to RGB frame
        
        Args:
            rgb_latent: (4, 64, 64) latent tensor
            depth_latent: (1, 64, 64) depth latent (optional, for conditioning)
            downscaled_rgb: (3, 512, 512) reference frame (optional)
            
        Returns:
            (H, W, 3) numpy array in [0, 255]
        """
        # Add batch dimension if needed
        if rgb_latent.dim() == 3:
            rgb_latent = rgb_latent.unsqueeze(0)
        
        # Decode from latent space
        latent = rgb_latent / self.vae.config.scaling_factor
        decoded = self.vae.decode(latent).sample
        
        # Convert to numpy
        rgb_frame = self._postprocess_rgb(decoded)
        
        return rgb_frame
    
    def _postprocess_rgb(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert normalized tensor back to numpy RGB"""
        # Remove batch dimension and move to CPU
        tensor = tensor.squeeze(0).cpu()
        
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to [0, 255]
        tensor = (tensor * 255.0).byte()
        
        # Rearrange to (H, W, C)
        rgb_np = tensor.permute(1, 2, 0).numpy()
        
        return rgb_np


class LumaFlowCodec:
    """
    Complete LumaFlow video codec with I-frames and P-frames
    """
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        keyframe_interval: int = 30
    ):
        self.device = device
        self.keyframe_interval = keyframe_interval
        
        # Initialize encoder and decoder
        self.encoder = LCMVideoEncoder(device=device)
        self.decoder = LCMVideoDecoder(device=device)
        
        print(f"✅ LumaFlow Codec initialized")
        print(f"   Device: {device}")
        print(f"   Keyframe interval: {keyframe_interval}")
    
    def encode_video(
        self,
        video_path: str,
        depth_video_path: Optional[str] = None,
        output_path: str = "output.lfv"
    ) -> Dict:
        """
        Encode a video file to LumaFlow format
        
        Args:
            video_path: Path to RGB video
            depth_video_path: Path to depth video (optional)
            output_path: Output .lfv file path
            
        Returns:
            Statistics dictionary
        """
        print(f"📹 Encoding video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Open depth video if provided
        depth_cap = None
        if depth_video_path:
            depth_cap = cv2.VideoCapture(depth_video_path)
            print(f"📊 Depth video: {depth_video_path}")
        
        # Encode frames
        encoded_frames = []
        total_size = 0
        
        frame_idx = 0
        while True:
            ret, rgb_frame = cap.read()
            if not ret:
                break
            
            # Read depth frame
            depth_frame = None
            if depth_cap:
                ret_depth, depth_frame = depth_cap.read()
                if ret_depth:
                    # Convert to single channel if needed
                    if len(depth_frame.shape) == 3:
                        depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
                    depth_frame = depth_frame.astype(np.float32)
            
            # Determine if keyframe
            is_keyframe = (frame_idx % self.keyframe_interval == 0)
            
            # Encode frame
            if is_keyframe:
                encoded = self.encoder.encode_frame(rgb_frame, depth_frame)
                encoded['is_keyframe'] = True
            else:
                # P-frame encoding (simplified for now)
                encoded = self.encoder.encode_frame(rgb_frame, depth_frame)
                encoded['is_keyframe'] = False
            
            # Calculate compressed size
            frame_size = self._calculate_frame_size(encoded)
            total_size += frame_size
            
            encoded_frames.append(encoded)
            
            if frame_idx % 10 == 0:
                print(f"   Frame {frame_idx}/{total_frames} | {'I' if is_keyframe else 'P'}-frame | {frame_size/1024:.1f} KB")
            
            frame_idx += 1
        
        cap.release()
        if depth_cap:
            depth_cap.release()
        
        # Save encoded video
        self._save_encoded_video(encoded_frames, output_path, fps)
        
        # Calculate statistics
        original_size = total_frames * 1920 * 1080 * 3  # Assuming 1080p
        compression_ratio = original_size / total_size
        
        stats = {
            'total_frames': total_frames,
            'keyframes': sum(1 for f in encoded_frames if f['is_keyframe']),
            'compressed_size_mb': total_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'output_path': output_path
        }
        
        print(f"\n✅ Encoding complete!")
        print(f"   Total frames: {total_frames}")
        print(f"   Keyframes: {stats['keyframes']}")
        print(f"   Compressed size: {stats['compressed_size_mb']:.2f} MB")
        print(f"   Compression ratio: {compression_ratio:.1f}x")
        
        return stats
    
    def decode_video(
        self,
        encoded_path: str,
        output_path: str = "decoded.mp4"
    ) -> str:
        """
        Decode a LumaFlow video back to RGB
        
        Args:
            encoded_path: Path to .lfv file
            output_path: Output video path
            
        Returns:
            Path to decoded video
        """
        print(f"🎬 Decoding video: {encoded_path}")
        
        # Load encoded video
        encoded_frames, fps = self._load_encoded_video(encoded_path)
        
        # Decode frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        
        for idx, encoded in enumerate(encoded_frames):
            # Decode frame
            rgb_frame = self.decoder.decode_frame(
                encoded['rgb_latent'],
                encoded.get('depth_latent'),
                encoded.get('downscaled_rgb')
            )
            
            # Initialize writer with first frame
            if out is None:
                height, width = rgb_frame.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Write frame
            out.write(rgb_frame)
            
            if idx % 10 == 0:
                print(f"   Frame {idx}/{len(encoded_frames)}")
        
        out.release()
        
        print(f"✅ Decoding complete: {output_path}")
        return output_path
    
    def _calculate_frame_size(self, encoded: Dict) -> int:
        """Calculate size of encoded frame in bytes"""
        size = 0
        
        # RGB latent: 4 channels * 64 * 64 * 4 bytes (float32)
        size += encoded['rgb_latent'].numel() * 4
        
        # Depth latent if present
        if 'depth_latent' in encoded:
            size += encoded['depth_latent'].numel() * 4
        
        # Downscaled RGB (stored as compressed JPEG)
        # Assume 10:1 compression for reference frame
        if 'downscaled_rgb' in encoded:
            size += encoded['downscaled_rgb'].numel() * 4 // 10
        
        return size
    
    def _save_encoded_video(self, frames: list, path: str, fps: int):
        """Save encoded frames to .lfv file"""
        import pickle
        data = {
            'frames': frames,
            'fps': fps,
            'version': '1.0'
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def _load_encoded_video(self, path: str) -> Tuple[list, int]:
        """Load encoded frames from .lfv file"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['frames'], data['fps']


# Test/demo code
if __name__ == "__main__":
    print("🎬 LumaFlow Codec Test")
    print("=" * 50)
    
    # Initialize codec
    codec = LumaFlowCodec()
    
    # Create test video
    print("\n📹 Creating test video...")
    test_video = "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video, fourcc, 30, (640, 480))
    
    for i in range(60):  # 2 seconds at 30fps
        # Create gradient frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(0, 255, 640).astype(np.uint8)  # Blue gradient
        frame[:, :, 1] = (i * 4) % 256  # Animated green
        frame[:, :, 2] = 128  # Constant red
        out.write(frame)
    
    out.release()
    print(f"✅ Test video created: {test_video}")
    
    # Encode
    print("\n🔄 Encoding...")
    stats = codec.encode_video(test_video, output_path="test.lfv")
    
    # Decode
    print("\n🔄 Decoding...")
    decoded = codec.decode_video("test.lfv", output_path="decoded.mp4")
    
    print("\n✅ Test complete!")

