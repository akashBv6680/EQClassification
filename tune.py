"""
Part 2 Option C: Hyperparameter Tuning for CIFAR-10
Performs grid search and ablation across:
  - Learning Rate: [1e-3, 3e-4]
  - Batch Size: [64, 128]
  - Optimizer: [AdamW, SGD + Momentum]
  - Dropout Rate: [0.2, 0.4]
"""

import argparse
import itertools
import torch
import torch.nn as nn
from models.baseline_cnn import BaselineCNNPyTorch
from models.resnet_cifar import ResNetCIFARPyTorch
from utils.dataset import get_pytorch_dataloaders

def run_experiment(model_type: str, lr: float, batch_size: int, opt_name: str, dropout: float, epochs: int = 5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_pytorch_dataloaders(batch_size=batch_size)

    if model_type == 'resnet':
        model = ResNetCIFARPyTorch(num_classes=10, dropout_rate=dropout).to(device)
    else:
        model = BaselineCNNPyTorch(num_classes=10, dropout_rate=dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    if opt_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

        val_acc = 100.0 * correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning Grid Search")
    parser.add_argument('--model', type=str, default='resnet', choices=['baseline', 'resnet'])
    parser.add_argument('--epochs-per-trial', type=int, default=5)
    args = parser.parse_args()

    param_grid = {
        'lr': [0.001, 0.0003],
        'batch_size': [64, 128],
        'optimizer': ['AdamW', 'SGD'],
        'dropout': [0.2, 0.4]
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"[*] Starting Hyperparameter Tuning Grid Search ({len(combinations)} trials)...")
    results = []
    for i, params in enumerate(combinations, 1):
        print(f"\n--- Trial {i}/{len(combinations)}: {params} ---")
        acc = run_experiment(args.model, params['lr'], params['batch_size'], params['optimizer'], params['dropout'], epochs=args.epochs_per_trial)
        results.append((params, acc))
        print(f"[+] Trial {i} Result: Best Val Acc = {acc:.2f}%")

    print("\n================ TUNING LEADERBOARD ================")
    results.sort(key=lambda x: x[1], reverse=True)
    for rank, (p, acc) in enumerate(results, 1):
        print(f"Rank {rank}: Val Acc = {acc:.2f}% | Params: {p}")

if __name__ == '__main__':
    main()
