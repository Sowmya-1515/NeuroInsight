"""
PHASE II - STEP 3: Enhanced Training with Balanced Sampling and Uncertainty Weighting
=====================================================================================
Features:
- Balanced sampling using weighted sampler
- Enhanced multi-task loss with uncertainty weighting
- Focal loss for severity classification
- Learning rate scheduling with warmup
- Gradient clipping for stability
- Comprehensive logging and checkpointing
- Early stopping to prevent overfitting
- MPS-compatible (all tensors float32)
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Import from enhanced step2_model
from step2_model_s import (
    Phase2Dataset, 
    create_model,
    EnhancedMultiTaskLoss,
    create_weighted_sampler,
    get_class_weights,
    get_optimizer,
    train_transform, 
    val_transform
)

# ============================================================
# CONFIG
# ============================================================
CSV_PATH      = "/Users/sowmyaalamuri/Desktop/Capstone_project/Phase2/phase2_dataset_enhanced.csv"
OUTPUT_DIR    = "/Users/sowmyaalamuri/Desktop/Capstone_project/Phase2/training_output_enhanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training hyperparameters
NUM_EPOCHS    = 25
BATCH_SIZE    = 64
LR            = 1e-4
EMBED_DIM     = 128
NUM_WORKERS   = 0  # Mac MPS works better with 0
FREEZE_EPOCHS = 3
SAVE_EVERY    = 5
WARMUP_EPOCHS = 2
EARLY_STOPPING_PATIENCE = 7

# ============================================================
# DEVICE
# ============================================================
def get_device():
    """Get the best available device (MPS-compatible)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✓ Using Apple MPS (Metal Performance Shaders)")
        # Set default dtype to float32 for MPS
        torch.set_default_dtype(torch.float32)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"⚠ Using CPU (training will be slow)")
    return device

device = get_device()


# ============================================================
# HELPER FUNCTIONS (MPS-COMPATIBLE)
# ============================================================
def move_batch(batch, device):
    """Move batch tensors to device with explicit float32 conversion"""
    for k in ["retrieval_label", "grade_label", "severity_label",
              "size_label", "location_label", "tumor_label"]:
        if k in batch:
            # Ensure labels are long tensors (MPS supports int64 for labels)
            batch[k] = batch[k].to(device, dtype=torch.long)
    
    if "sample_weight" in batch:
        # Sample weights must be float32 for MPS
        batch["sample_weight"] = batch["sample_weight"].to(device, dtype=torch.float32)
    
    return batch


def compute_accs(outputs, batch):
    """Compute accuracy metrics for all tasks"""
    mask = batch["tumor_label"] == 1

    def masked_acc(logits, labels):
        if mask.sum() == 0:
            return 0.0
        # Ensure logits and labels are on same device
        return (logits[mask].argmax(1) == labels[mask]).float().mean().item()

    # Tumor accuracy (all slices)
    tumor_acc = (outputs["tumor_logits"].argmax(1) == batch["tumor_label"]).float().mean().item()
    
    # Classification accuracies (only tumor slices)
    grade_acc = masked_acc(outputs["grade_logits"], batch["grade_label"])
    severity_acc = masked_acc(outputs["severity_logits"], batch["severity_label"])
    size_acc = masked_acc(outputs["size_logits"], batch["size_label"])
    location_acc = masked_acc(outputs["location_logits"], batch["location_label"])
    
    return {
        "tumor": tumor_acc,
        "grade": grade_acc,
        "severity": severity_acc,
        "size": size_acc,
        "location": location_acc,
    }


def compute_loss_weights(model, val_loader, criterion, device):
    """Compute task weights for logging"""
    model.eval()
    task_losses = {k: 0.0 for k in ['supcon', 'ntxent', 'grade', 'severity', 
                                     'size', 'location', 'tumor']}
    n_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, dtype=torch.float32)
            batch = move_batch(batch, device)
            
            outputs = model(images)
            losses = criterion(outputs, batch)
            
            for k in task_losses:
                if k in losses:
                    task_losses[k] += losses[k]
            n_batches += 1
    
    # Average
    for k in task_losses:
        task_losses[k] /= n_batches
    
    return task_losses


def run_epoch(model, loader, loss_fn, optimizer, device, training=True, epoch=0):
    """
    Run one training or validation epoch (MPS-optimized)
    
    Args:
        model: The model
        loader: DataLoader
        loss_fn: Loss function
        optimizer: Optimizer (None for validation)
        device: Torch device
        training: True for training, False for validation
        epoch: Current epoch number (for logging)
    """
    if training:
        model.train()
        desc = "  Train"
    else:
        model.eval()
        desc = "  Val"

    total_loss = 0.0
    accs = {"tumor": 0.0, "grade": 0.0, "severity": 0.0, "size": 0.0, "location": 0.0}
    task_losses = {k: 0.0 for k in ['supcon', 'ntxent', 'grade', 'severity', 
                                     'size', 'location', 'tumor']}
    n = 0
    task_weights_sum = {}

    ctx = torch.enable_grad if training else torch.no_grad
    with ctx():
        pbar = tqdm(loader, desc=desc, leave=False)
        for batch_idx, batch in enumerate(pbar):
            # Move images with explicit float32
            images = batch["image"].to(device, dtype=torch.float32)
            batch = move_batch(batch, device)

            if training:
                optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            losses = loss_fn(outputs, batch)

            if training:
                losses["total"].backward()
                # Gradient clipping (works with MPS)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bsz = images.size(0)
            total_loss += losses["total"].item() * bsz
            
            # Accumulate task losses
            for k in task_losses:
                if k in losses:
                    task_losses[k] += losses[k] * bsz
            
            # Accumulate task weights
            if "task_weights" in losses:
                for k, v in losses["task_weights"].items():
                    if k not in task_weights_sum:
                        task_weights_sum[k] = 0.0
                    task_weights_sum[k] += v * bsz
            
            # Compute accuracies
            batch_accs = compute_accs(outputs, batch)
            for k in accs:
                accs[k] += batch_accs[k] * bsz
                
            n += bsz
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'grade': f"{batch_accs['grade']:.3f}",
                'sev': f"{batch_accs['severity']:.3f}"
            })

    # Average metrics
    results = {
        "loss": total_loss / n,
        **{k: v / n for k, v in accs.items()}
    }
    
    # Average task losses
    results["task_losses"] = {k: v / n for k, v in task_losses.items()}
    
    # Average task weights
    if task_weights_sum:
        results["task_weights"] = {k: v / n for k, v in task_weights_sum.items()}
    
    return results


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def plot_training_history(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#0d0d0d')
    
    epochs = [h['epoch'] for h in history]
    
    # Loss plot
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    ax.plot(epochs, [h['train']['loss'] for h in history], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, [h['val']['loss'] for h in history], 'r-', label='Val', linewidth=2)
    ax.set_title('Total Loss', color='white', fontsize=12)
    ax.set_xlabel('Epoch', color='#aaa')
    ax.set_ylabel('Loss', color='#aaa')
    ax.legend(facecolor='#1a1a1a', labelcolor='white')
    ax.tick_params(colors='#aaa')
    
    # Grade accuracy
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    ax.plot(epochs, [h['train']['grade'] for h in history], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, [h['val']['grade'] for h in history], 'r-', label='Val', linewidth=2)
    ax.set_title('Grade Accuracy', color='white', fontsize=12)
    ax.set_xlabel('Epoch', color='#aaa')
    ax.set_ylabel('Accuracy', color='#aaa')
    ax.legend(facecolor='#1a1a1a', labelcolor='white')
    ax.tick_params(colors='#aaa')
    
    # Severity accuracy
    ax = axes[0, 2]
    ax.set_facecolor('#1a1a1a')
    ax.plot(epochs, [h['train']['severity'] for h in history], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, [h['val']['severity'] for h in history], 'r-', label='Val', linewidth=2)
    ax.set_title('Severity Accuracy', color='white', fontsize=12)
    ax.set_xlabel('Epoch', color='#aaa')
    ax.set_ylabel('Accuracy', color='#aaa')
    ax.legend(facecolor='#1a1a1a', labelcolor='white')
    ax.tick_params(colors='#aaa')
    
    # Size accuracy
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    ax.plot(epochs, [h['train']['size'] for h in history], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, [h['val']['size'] for h in history], 'r-', label='Val', linewidth=2)
    ax.set_title('Size Accuracy', color='white', fontsize=12)
    ax.set_xlabel('Epoch', color='#aaa')
    ax.set_ylabel('Accuracy', color='#aaa')
    ax.legend(facecolor='#1a1a1a', labelcolor='white')
    ax.tick_params(colors='#aaa')
    
    # Location accuracy
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.plot(epochs, [h['train']['location'] for h in history], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, [h['val']['location'] for h in history], 'r-', label='Val', linewidth=2)
    ax.set_title('Location Accuracy', color='white', fontsize=12)
    ax.set_xlabel('Epoch', color='#aaa')
    ax.set_ylabel('Accuracy', color='#aaa')
    ax.legend(facecolor='#1a1a1a', labelcolor='white')
    ax.tick_params(colors='#aaa')
    
    # Tumor accuracy
    ax = axes[1, 2]
    ax.set_facecolor('#1a1a1a')
    ax.plot(epochs, [h['train']['tumor'] for h in history], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, [h['val']['tumor'] for h in history], 'r-', label='Val', linewidth=2)
    ax.set_title('Tumor Detection Accuracy', color='white', fontsize=12)
    ax.set_xlabel('Epoch', color='#aaa')
    ax.set_ylabel('Accuracy', color='#aaa')
    ax.legend(facecolor='#1a1a1a', labelcolor='white')
    ax.tick_params(colors='#aaa')
    
    plt.suptitle('Training History', color='white', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close()
    print(f"  ✓ Training plot saved: {save_path}")


def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, 
                   history, is_best=False, filename=None):
    """Save training checkpoint"""
    if filename is None:
        filename = "best_model.pth" if is_best else f"checkpoint_epoch{epoch}.pth"
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'history': history,
    }
    
    path = os.path.join(OUTPUT_DIR, filename)
    torch.save(checkpoint, path)
    print(f"  ✓ Checkpoint saved: {path}")


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================
def main():
    print("\n" + "="*70)
    print("  PHASE II - STEP 3: Enhanced Multi-Task Training")
    print("  With Balanced Sampling + Focal Loss + Uncertainty Weighting")
    print("="*70)

    # Load enhanced dataset
    print(f"\n📂 Loading enhanced dataset from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {len(df):,} rows")
    print(f"  Tumor slices: {(df['tumor_present']==1).sum():,} ({(df['tumor_present']==1).mean()*100:.1f}%)")
    
    # Check for sample weights
    if 'sample_weight' in df.columns:
        print(f"  ✓ Sample weights found - min: {df['sample_weight'].min():.3f}, "
              f"max: {df['sample_weight'].max():.3f}")
    else:
        print(f"  ⚠ No sample weights found - will use uniform weights")

    # Split data
    print(f"\n📊 Splitting data (80/20 train/val)...")
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["tumor_present"], random_state=42
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"  Train: {len(train_df):,} slices")
    print(f"  Val:   {len(val_df):,} slices")

    # Create datasets
    print(f"\n🔧 Creating datasets...")
    print("  Train dataset (with augmentation):")
    train_dataset = Phase2Dataset(train_df, transform=train_transform, use_sample_weights=True)
    
    print("\n  Val dataset (no augmentation):")
    val_dataset = Phase2Dataset(val_df, transform=val_transform, use_sample_weights=True)

    # Create weighted sampler for balanced training
    print(f"\n⚖️  Creating weighted sampler...")
    train_sampler = create_weighted_sampler(train_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # Get class weights
    print(f"\n⚖️  Computing class weights...")
    weights = get_class_weights(df, device)

    # Create model
    print(f"\n🏗️  Creating enhanced multi-task model...")
    model = create_model(embedding_dim=EMBED_DIM, pretrained=True, device=device)

    # Create loss function
    print(f"\n📉 Creating enhanced loss function...")
    loss_fn = EnhancedMultiTaskLoss(
        grade_weight=weights.get('grade'),
        severity_weight=weights.get('severity'),
        size_weight=weights.get('size'),
        location_weight=weights.get('location'),
        tumor_weight=weights.get('tumor'),
        device=device
    )

    # Initial training with frozen backbone
    print(f"\n❄️  Freezing backbone for first {FREEZE_EPOCHS} epochs...")
    model.freeze_backbone()
    optimizer = get_optimizer(model, LR, frozen=True)
    
    # Learning rate scheduler with warmup
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=NUM_EPOCHS,
        base_lr=LR,
        min_lr=1e-6
    )

    # Training tracking
    best_val_loss = float("inf")
    best_val_acc = 0.0
    history = []
    start_epoch = 1
    epochs_without_improvement = 0

    # Check for existing checkpoint
    checkpoint_path = os.path.join(OUTPUT_DIR, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        user = input("\n📦 Resume from checkpoint? (y/n): ").strip().lower()
        if user == "y":
            print("  Loading checkpoint...")
            ckpt = torch.load(checkpoint_path, map_location=device)
            try:
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt["epoch"] + 1
                best_val_loss = ckpt["best_val_loss"]
                history = ckpt.get("history", [])
                
                # Handle unfreezing if needed
                if start_epoch > FREEZE_EPOCHS + 1:
                    model.unfreeze_backbone()
                    optimizer = get_optimizer(model, LR, frozen=False)
                    scheduler = WarmupCosineScheduler(
                        optimizer, 
                        warmup_epochs=WARMUP_EPOCHS,
                        total_epochs=NUM_EPOCHS,
                        base_lr=LR,
                        min_lr=1e-6
                    )
                
                print(f"  ✓ Resumed from epoch {start_epoch - 1}")
            except Exception as e:
                print(f"  ⚠ Could not load checkpoint ({e}) — starting fresh.")
                start_epoch = 1
        else:
            print("  Starting fresh training.")
    else:
        print("\n  No checkpoint found - starting fresh training.")

    # Training loop
    print(f"\n{'='*70}")
    print(f"  Starting Training for {NUM_EPOCHS} epochs")
    print(f"{'='*70}")
    
    total_start_time = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        
        # Unfreeze backbone after FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS + 1:
            print(f"\n🔥 Unfreezing backbone for fine-tuning...")
            model.unfreeze_backbone()
            optimizer = get_optimizer(model, LR, frozen=False)
            scheduler = WarmupCosineScheduler(
                optimizer, 
                warmup_epochs=WARMUP_EPOCHS,
                total_epochs=NUM_EPOCHS,
                base_lr=LR,
                min_lr=1e-6
            )

        # Get current learning rate
        current_lr = scheduler.step()
        
        # Print epoch header
        frozen_status = "[FROZEN]" if epoch <= FREEZE_EPOCHS else "[ACTIVE]"
        print(f"\n{'='*70}")
        print(f"  Epoch {epoch}/{NUM_EPOCHS} {frozen_status}  LR={current_lr:.2e}")
        print(f"{'='*70}")

        # Training
        train_results = run_epoch(
            model, train_loader, loss_fn, optimizer, device, 
            training=True, epoch=epoch
        )

        # Validation
        val_results = run_epoch(
            model, val_loader, loss_fn, None, device, 
            training=False, epoch=epoch
        )

        # Print results
        print(f"\n  📈 Training Results:")
        print(f"    Loss: {train_results['loss']:.4f}")
        print(f"    Grade: {train_results['grade']:.4f} | Severity: {train_results['severity']:.4f}")
        print(f"    Size: {train_results['size']:.4f} | Location: {train_results['location']:.4f}")
        print(f"    Tumor: {train_results['tumor']:.4f}")
        
        print(f"\n  📊 Validation Results:")
        print(f"    Loss: {val_results['loss']:.4f}")
        print(f"    Grade: {val_results['grade']:.4f} | Severity: {val_results['severity']:.4f}")
        print(f"    Size: {val_results['size']:.4f} | Location: {val_results['location']:.4f}")
        print(f"    Tumor: {val_results['tumor']:.4f}")
        
        if "task_weights" in val_results:
            print(f"\n  ⚖️  Task Weights:")
            for k, v in val_results["task_weights"].items():
                print(f"    {k}: {v:.4f}")

        epoch_time = time.time() - epoch_start_time
        print(f"\n  ⏱️  Epoch time: {epoch_time:.1f}s")

        # Save best model
        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]
            best_val_acc = val_results["grade"]  # Track best grade accuracy
            epochs_without_improvement = 0
            save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, 
                          history, is_best=True)
            print(f"  🏆 New best model! (val_loss={best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs")

        # Save periodic checkpoint
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, 
                          history, is_best=False, filename=f"checkpoint_epoch{epoch}_s.pth")

        # Save latest checkpoint
        save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, 
                      history, is_best=False, filename="latest_checkpoint_s.pth")

        # Update history
        history.append({
            "epoch": epoch,
            "train": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                     for k, v in train_results.items()},
            "val": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in val_results.items()},
            "lr": float(current_lr),
            "time": float(epoch_time)
        })

        # Early stopping check
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\n🛑 Early stopping triggered after {epoch} epochs")
            break

        # Print separator
        print(f"\n{'-'*70}")

    # Training complete
    total_time = time.time() - total_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    print(f"\n{'='*70}")
    print(f"  ✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n  Total training time: {hours}h {minutes}m")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best validation grade accuracy: {best_val_acc:.4f}")
    print(f"  Total epochs trained: {len(history)}")
    
    # Save final history
    history_path = os.path.join(OUTPUT_DIR, "training_history_s.json")
    with open(history_path, "w") as f:
        # Convert numpy values to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.item()
            return obj
        
        json.dump(history, f, indent=2, default=convert)
    print(f"  ✓ History saved: {history_path}")
    
    # Plot training curves
    plot_path = os.path.join(OUTPUT_DIR, "training_curve_s.png")
    plot_training_history(history, plot_path)
    
    # Final evaluation on best model
    print(f"\n📋 Loading best model for final evaluation...")
    best_checkpoint = torch.load(os.path.join(OUTPUT_DIR, "best_model_s.pth"), 
                                 map_location=device)
    model.load_state_dict(best_checkpoint)
    model.eval()
    
    # Run final validation
    print(f"\n🔍 Running final validation on best model...")
    final_results = run_epoch(
        model, val_loader, loss_fn, None, device, 
        training=False, epoch=NUM_EPOCHS
    )
    
    print(f"\n{'='*70}")
    print(f"  FINAL MODEL PERFORMANCE")
    print(f"{'='*70}")
    print(f"\n  Validation Loss: {final_results['loss']:.4f}")
    print(f"\n  Classification Accuracies:")
    print(f"    Grade    : {final_results['grade']*100:.2f}%")
    print(f"    Severity : {final_results['severity']*100:.2f}%")
    print(f"    Size     : {final_results['size']*100:.2f}%")
    print(f"    Location : {final_results['location']*100:.2f}%")
    print(f"    Tumor    : {final_results['tumor']*100:.2f}%")
    
    # Compute average accuracy
    avg_acc = (final_results['grade'] + final_results['severity'] + 
               final_results['size'] + final_results['location']) / 4 * 100
    print(f"\n  Average Classification Accuracy: {avg_acc:.2f}%")
    
    print(f"\n{'='*70}")
    print(f"  ✅ STEP 3 COMPLETE!")
    print(f"  Outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")
    
    return history, final_results


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    try:
        history, final_results = main()
    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted by user. Saving checkpoint...")
        # Save emergency checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'history': history
        }, os.path.join(OUTPUT_DIR, "interrupted_checkpoint_s.pth"))
        print("  ✓ Emergency checkpoint saved")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()