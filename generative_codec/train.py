"""
Training script for LumaFlow codec
Fine-tunes LCM encoder/decoder on iPhone LiDAR data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import time

from models.lcm_codec import LCMVideoEncoder, LCMVideoDecoder
from data.iphone_loader import create_iphone_dataloader


class LumaFlowTrainer:
    """Trainer for LumaFlow codec"""
    
    def __init__(
        self,
        encoder: LCMVideoEncoder,
        decoder: LCMVideoDecoder,
        train_loader,
        val_loader,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        log_dir: str = "runs/lumaflow"
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Optimizer (only train decoder initially, encoder is frozen)
        trainable_params = list(self.decoder.parameters())
        self.optimizer = optim.AdamW(trainable_params, lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=50,
            eta_min=1e-6
        )
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        
        print(f"✅ Trainer initialized")
        print(f"   Device: {device}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch"""
        self.decoder.train()
        self.encoder.eval()  # Keep encoder frozen
        
        total_loss = 0
        total_mse = 0
        total_l1 = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            rgb = batch['rgb'].to(self.device)
            depth = batch['depth'].to(self.device)
            
            # Encode (no gradients)
            with torch.no_grad():
                # Encode each frame in batch
                latents = []
                for i in range(rgb.size(0)):
                    rgb_np = (rgb[i].cpu().permute(1, 2, 0).numpy() + 1) * 127.5
                    rgb_np = rgb_np.astype('uint8')
                    
                    depth_np = depth[i, 0].cpu().numpy()
                    
                    encoded = self.encoder.encode_frame(rgb_np, depth_np)
                    latents.append(encoded['rgb_latent'])
                
                latent_batch = torch.stack(latents)
            
            # Decode
            self.optimizer.zero_grad()
            
            # Decode latents back to RGB
            decoded_batch = []
            for i in range(latent_batch.size(0)):
                decoded_np = self.decoder.decode_frame(latent_batch[i])
                
                # Convert back to tensor
                decoded_tensor = torch.from_numpy(decoded_np).float() / 127.5 - 1.0
                decoded_tensor = decoded_tensor.permute(2, 0, 1)
                decoded_batch.append(decoded_tensor)
            
            decoded = torch.stack(decoded_batch).to(self.device)
            
            # Calculate losses
            mse = self.mse_loss(decoded, rgb)
            l1 = self.l1_loss(decoded, rgb)
            
            # Combined loss
            loss = mse + 0.1 * l1
            
            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_mse += mse.item()
            total_l1 += l1.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{mse.item():.4f}',
                'psnr': f'{self._mse_to_psnr(mse.item()):.2f}'
            })
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/mse', mse.item(), self.global_step)
                self.writer.add_scalar('train/psnr', self._mse_to_psnr(mse.item()), self.global_step)
            
            self.global_step += 1
        
        # Epoch metrics
        num_batches = len(self.train_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'l1': total_l1 / num_batches,
            'psnr': self._mse_to_psnr(total_mse / num_batches)
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validate the model"""
        self.decoder.eval()
        
        total_mse = 0
        total_l1 = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            rgb = batch['rgb'].to(self.device)
            depth = batch['depth'].to(self.device)
            
            # Encode
            latents = []
            for i in range(rgb.size(0)):
                rgb_np = (rgb[i].cpu().permute(1, 2, 0).numpy() + 1) * 127.5
                rgb_np = rgb_np.astype('uint8')
                depth_np = depth[i, 0].cpu().numpy()
                
                encoded = self.encoder.encode_frame(rgb_np, depth_np)
                latents.append(encoded['rgb_latent'])
            
            latent_batch = torch.stack(latents)
            
            # Decode
            decoded_batch = []
            for i in range(latent_batch.size(0)):
                decoded_np = self.decoder.decode_frame(latent_batch[i])
                decoded_tensor = torch.from_numpy(decoded_np).float() / 127.5 - 1.0
                decoded_tensor = decoded_tensor.permute(2, 0, 1)
                decoded_batch.append(decoded_tensor)
            
            decoded = torch.stack(decoded_batch).to(self.device)
            
            # Calculate losses
            mse = self.mse_loss(decoded, rgb)
            l1 = self.l1_loss(decoded, rgb)
            
            total_mse += mse.item()
            total_l1 += l1.item()
        
        # Metrics
        num_batches = len(self.val_loader)
        metrics = {
            'mse': total_mse / num_batches,
            'l1': total_l1 / num_batches,
            'psnr': self._mse_to_psnr(total_mse / num_batches)
        }
        
        # Log to tensorboard
        self.writer.add_scalar('val/mse', metrics['mse'], epoch)
        self.writer.add_scalar('val/psnr', metrics['psnr'], epoch)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: dict, path: str):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }, path)
        print(f"💾 Checkpoint saved: {path}")
    
    def _mse_to_psnr(self, mse: float) -> float:
        """Convert MSE to PSNR (dB)"""
        if mse == 0:
            return 100.0
        # Assuming pixel values in [0, 1] after normalization
        return 10 * torch.log10(torch.tensor(4.0 / mse)).item()


def main():
    parser = argparse.ArgumentParser(description="Train LumaFlow codec")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with iPhone .mov files')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='runs/lumaflow', help='Tensorboard log directory')
    
    args = parser.parse_args()
    
    print("🎬 LumaFlow Codec Training")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print("=" * 50)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize models
    print("\n📦 Initializing models...")
    encoder = LCMVideoEncoder(device=args.device)
    decoder = LCMVideoDecoder(device=args.device)
    
    # Create dataloaders
    print("\n📂 Loading data...")
    train_loader = create_iphone_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Simple validation split (use last 10% of data)
    # In production, you'd want proper train/val split
    val_loader = create_iphone_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        max_frames_per_video=int(100 * args.val_split)
    )
    
    # Initialize trainer
    print("\n🏋️ Initializing trainer...")
    trainer = LumaFlowTrainer(
        encoder=encoder,
        decoder=decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.lr,
        log_dir=args.log_dir
    )
    
    # Training loop
    print("\n🚀 Starting training...")
    best_psnr = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # Train
        train_metrics = trainer.train_epoch(epoch)
        
        # Validate
        val_metrics = trainer.validate(epoch)
        
        # Update learning rate
        trainer.scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Print metrics
        print(f"\n📊 Epoch {epoch} Results:")
        print(f"   Train Loss: {train_metrics['loss']:.4f}")
        print(f"   Train PSNR: {train_metrics['psnr']:.2f} dB")
        print(f"   Val PSNR: {val_metrics['psnr']:.2f} dB")
        print(f"   Time: {epoch_time:.1f}s")
        
        # Save checkpoint
        if epoch % 5 == 0 or val_metrics['psnr'] > best_psnr:
            checkpoint_path = checkpoint_dir / f"lumaflow_epoch{epoch}.pt"
            trainer.save_checkpoint(epoch, val_metrics, str(checkpoint_path))
            
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                best_path = checkpoint_dir / "lumaflow_best.pt"
                trainer.save_checkpoint(epoch, val_metrics, str(best_path))
                print(f"🎯 New best PSNR: {best_psnr:.2f} dB")
    
    print("\n✅ Training complete!")
    print(f"Best validation PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    # Close tensorboard writer
    trainer.writer.close()


if __name__ == "__main__":
    main()

