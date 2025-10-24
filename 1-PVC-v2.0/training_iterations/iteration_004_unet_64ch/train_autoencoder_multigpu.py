#!/usr/bin/env python3
"""
Train True Autoencoder with Multi-GPU (DDP)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import sys
import os
from pathlib import Path
import time
import datetime

sys.path.insert(0, str(Path(__file__).parent))
from models.true_autoencoder import CompressionAutoencoder, PerceptualLoss


class PreloadedAnimeDataset(Dataset):
    def __init__(self, npy_path, load_to_ram=False):  # Changed default to False (use mmap)
        print(f"Loading dataset from {npy_path}...")
        if load_to_ram:
            print("Loading entire dataset to RAM (this may take a minute)...")
            self.data = np.load(npy_path)  # Load to RAM, not memory-mapped
            print(f"✓ Loaded {len(self.data)} frames to RAM, shape: {self.data.shape}")
        else:
            # Use memory-mapped mode to avoid loading 71GB × 8 workers = 568GB
            self.data = np.load(npy_path, mmap_mode='r')
            print(f"✓ Loaded {len(self.data)} frames (memory-mapped), shape: {self.data.shape}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        frame = self.data[idx]  # (H, W, 3) uint8 in [0, 255]
        # IMPORTANT: .copy() to force load into RAM, avoiding mmap deadlocks
        # Convert to float and normalize to [0, 1]
        frame_tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
        return frame_tensor


def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    print(f"[Rank {rank}] Initializing process group on port {master_port}...", flush=True)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=10))
    print(f"[Rank {rank}] Process group initialized successfully!", flush=True)


def cleanup():
    dist.destroy_process_group()


def train_worker(rank, world_size, args):
    setup(rank, world_size, args['master_port'])
    
    if rank == 0:
        print("="*80, flush=True)
        print("TRAINING TRUE AUTOENCODER (MULTI-GPU)", flush=True)
        print("="*80, flush=True)
        print(f"GPUs: {world_size}", flush=True)
        print(f"Dataset: {args['dataset_path']}", flush=True)
        print(f"Epochs: {args['epochs']}", flush=True)
        print(f"Batch size per GPU: {args['batch_size']}", flush=True)
        print(f"Total batch size: {args['batch_size'] * world_size}", flush=True)
        print(flush=True)
    
    # Load dataset - CRITICAL: load_to_ram=True to avoid mmap deadlock in DDP
    print(f"[Rank {rank}] Loading dataset to RAM (this will take ~2 minutes)...", flush=True)
    dataset = PreloadedAnimeDataset(args['dataset_path'], load_to_ram=True)
    print(f"[Rank {rank}] Dataset loaded: {len(dataset)} frames", flush=True)
    
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    
    print(f"[Rank {rank}] Creating DataLoaders...", flush=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args['batch_size'], sampler=train_sampler,
        num_workers=0, pin_memory=True  # num_workers=0 to avoid multiprocessing deadlock with RAM dataset
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args['batch_size'], sampler=val_sampler,
        num_workers=0, pin_memory=True  # num_workers=0 to avoid multiprocessing deadlock
    )
    print(f"[Rank {rank}] DataLoaders created", flush=True)
    
    if rank == 0:
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)
        print(flush=True)
    
    # Create model
    print(f"[Rank {rank}] Creating model...", flush=True)
    model = CompressionAutoencoder(latent_channels=args['latent_channels']).to(rank)
    print(f"[Rank {rank}] Model created, wrapping with DDP...", flush=True)
    model = DDP(model, device_ids=[rank])
    print(f"[Rank {rank}] DDP model ready", flush=True)
    
    if rank == 0:
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")
        print()
    
    # Loss
    print(f"[Rank {rank}] Creating loss functions...", flush=True)
    mse_loss = nn.MSELoss()
    if args['use_perceptual']:
        perceptual_loss = PerceptualLoss().to(rank)
        if rank == 0:
            print("Using MSE + Perceptual loss", flush=True)
    else:
        perceptual_loss = None
        if rank == 0:
            print("Using MSE only", flush=True)
    
    if rank == 0:
        print(flush=True)
    
    # Optimizer
    print(f"[Rank {rank}] Creating optimizer...", flush=True)
    optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])
    print(f"[Rank {rank}] Optimizer created", flush=True)
    
    # Try to load checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = Path(args['output_dir']) / 'checkpoint_latest.pth'
    
    print(f"[Rank {rank}] Checking for checkpoint...", flush=True)
    if checkpoint_path.exists() and rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}...", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"✓ Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}", flush=True)
        print(flush=True)
    
    # Broadcast checkpoint data to all ranks
    print(f"[Rank {rank}] Broadcasting checkpoint info...", flush=True)
    if world_size > 1:
        start_epoch_tensor = torch.tensor(start_epoch).to(rank)
        best_val_loss_tensor = torch.tensor(best_val_loss).to(rank)
        dist.broadcast(start_epoch_tensor, src=0)
        dist.broadcast(best_val_loss_tensor, src=0)
        start_epoch = start_epoch_tensor.item()
        best_val_loss = best_val_loss_tensor.item()
    print(f"[Rank {rank}] Ready to train from epoch {start_epoch}", flush=True)
    
    for epoch in range(start_epoch, args['epochs']):
        if rank == 0:
            print(f"Starting Epoch {epoch+1}/{args['epochs']}...", flush=True)
        
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        
        # Train
        model.train()
        train_loss = 0
        
        if rank == 0:
            print(f"Entering training loop...", flush=True)
        
        batch_count = 0
        for images in train_loader:
            if rank == 0 and batch_count == 0:
                print(f"[Rank 0] Batch 0: Starting...", flush=True)
            
            images = images.to(rank)
            if rank == 0 and batch_count == 0:
                print(f"[Rank 0] Batch 0: Moved to GPU, shape={images.shape}", flush=True)
            
            # Handle size mismatch - crop to multiple of 32
            h, w = images.shape[2], images.shape[3]
            h_new = (h // 32) * 32
            w_new = (w // 32) * 32
            if h != h_new or w != w_new:
                images = images[:, :, :h_new, :w_new]
            
            if rank == 0 and batch_count == 0:
                print(f"[Rank 0] Batch 0: Cropped to {images.shape}, calling model.forward()...", flush=True)
            
            reconstructed, latent = model(images)
            
            if rank == 0 and batch_count == 0:
                print(f"[Rank 0] Batch 0: Forward complete! Calculating loss...", flush=True)
            
            loss_mse = mse_loss(reconstructed, images)
            
            if args['use_perceptual']:
                loss_perc = perceptual_loss(reconstructed, images)
                loss = 0.7 * loss_mse + 0.3 * loss_perc
            else:
                loss = loss_mse
            
            if rank == 0 and batch_count == 0:
                print(f"[Rank 0] Batch 0: Loss = {loss.item():.6f}, starting backward...", flush=True)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if rank == 0 and batch_count == 0:
                print(f"[Rank 0] Batch 0: Complete! ✓", flush=True)
            
            train_loss += loss.item()
            batch_count += 1
            
            # Log progress every 100 batches
            if rank == 0 and batch_count % 100 == 0:
                avg_loss = train_loss / batch_count
                progress_pct = (batch_count / len(train_loader)) * 100
                print(f"  Batch {batch_count}/{len(train_loader)} ({progress_pct:.1f}%) - Avg Loss: {avg_loss:.6f}", flush=True)
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_psnr = 0
        
        with torch.no_grad():
            for images in val_loader:
                images = images.to(rank)
                
                # Crop to multiple of 32
                h, w = images.shape[2], images.shape[3]
                h_new = (h // 32) * 32
                w_new = (w // 32) * 32
                if h != h_new or w != w_new:
                    images = images[:, :, :h_new, :w_new]
                
                reconstructed, latent = model(images)
                
                loss_mse = mse_loss(reconstructed, images)
                val_loss += loss_mse.item()
                
                psnr = 20 * np.log10(1.0 / np.sqrt(loss_mse.item())) if loss_mse.item() > 0 else 100
                val_psnr += psnr
        
        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        
        # Gather metrics from all GPUs
        val_loss_tensor = torch.tensor(val_loss).to(rank)
        val_psnr_tensor = torch.tensor(val_psnr).to(rank)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_psnr_tensor, op=dist.ReduceOp.SUM)
        val_loss = (val_loss_tensor / world_size).item()
        val_psnr = (val_psnr_tensor / world_size).item()
        
        scheduler.step()
        
        elapsed = time.time() - start_time
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{args['epochs']} ({elapsed:.1f}s):")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}, PSNR: {val_psnr:.2f} dB")
            
            # Save latest checkpoint (every epoch for crash recovery)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_psnr': val_psnr,
                'best_val_loss': best_val_loss,
            }, Path(args['output_dir']) / 'checkpoint_latest.pth')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_psnr': val_psnr,
                }, Path(args['output_dir']) / 'autoencoder_best.pth')
                print(f"  ✓ Saved best model")
            
            # Save epoch checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, Path(args['output_dir']) / f'autoencoder_epoch_{epoch+1}.pth')
                print(f"  ✓ Saved checkpoint epoch_{epoch+1}")
            
            print()
    
    if rank == 0:
        print("="*80)
        print(f"TRAINING COMPLETE - Best val loss: {best_val_loss:.6f}")
        print("="*80)
    
    cleanup()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./trained_models')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent-channels', type=int, default=32)
    parser.add_argument('--no-perceptual', action='store_true')
    
    args = parser.parse_args()
    
    # Get number of GPUs
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Package args
    train_args = {
        'dataset_path': args.dataset,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'latent_channels': args.latent_channels,
        'use_perceptual': not args.no_perceptual,
        'master_port': 29500 + np.random.randint(0, 1000),  # Random port, but shared across all workers
    }
    
    print(f"Using port: {train_args['master_port']}")
    
    # Launch multi-GPU training
    mp.spawn(train_worker, args=(world_size, train_args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

