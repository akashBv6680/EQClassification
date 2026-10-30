"""
CIFAR-10 Training Script
Supports both PyTorch and TensorFlow.
Includes:
  - Loss computation (CrossEntropy)
  - Accuracy metrics
  - Learning rate scheduling (Cosine Annealing)
  - Gradient clipping (Option B)
  - Mixed precision training (Optional Extension)
  - Checkpoint saving
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from models.baseline_cnn import BaselineCNNPyTorch, create_baseline_cnn_tf
from models.resnet_cifar import ResNetCIFARPyTorch, create_resnet_cifar_tf
from utils.dataset import get_pytorch_dataloaders, get_tf_datasets
from utils.callbacks import EarlyStoppingPyTorch, ModelCheckpointerPyTorch

def set_seed(seed: int = 42):
    """Ensure strict reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[*] Global random seed set to {seed}")

def train_pytorch(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training on device: {device}")

    # 1. Data Pipeline
    train_loader, test_loader = get_pytorch_dataloaders(batch_size=args.batch_size)

    # 2. Model Architecture
    if args.model == 'resnet':
        model = ResNetCIFARPyTorch(num_classes=10, dropout_rate=args.dropout).to(device)
    else:
        model = BaselineCNNPyTorch(num_classes=10, dropout_rate=args.dropout).to(device)

    # 3. Loss, Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=args.amp)

    # 4. Callbacks (Option B)
    early_stopper = EarlyStoppingPyTorch(patience=args.patience, checkpoint_path=f"checkpoints/best_{args.model}.pt")
    checkpointer = ModelCheckpointerPyTorch(filepath=f"checkpoints/{args.model}_acc.pt")

    # 5. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Gradient scaling for mixed-precision
            scaler.scale(loss).backward()

            # Gradient clipping (Option B)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast(enabled=args.amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total
        curr_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {curr_lr:.6f}")

        checkpointer.step(val_acc, model, epoch)
        if early_stopper(val_loss, model):
            break

    print(f"[*] Training finished. Optimal model weights saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CIFAR-10 classifier')
    parser.add_argument('--framework', type=str, default='pytorch', choices=['pytorch', 'tensorflow'])
    parser.add_argument('--model', type=str, default='resnet', choices=['baseline', 'resnet'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_pytorch(args)
