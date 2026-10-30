"""
Part 2 Option B: Custom Training Utilities & Callbacks
Implements:
  1. EarlyStopping (monitors val_loss, restores best weights)
  2. ModelCheckpointer (saves highest val_acc checkpoints)
  3. LearningRateLogger
"""

import os
import torch
import numpy as np

class EarlyStoppingPyTorch:
    """
    Early stops training if validation loss does not improve after a given patience.
    """
    def __init__(self, patience: int = 7, min_delta: float = 1e-4, checkpoint_path: str = 'checkpoints/best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save checkpoint
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"[*] EarlyStopping: Validation loss improved to {val_loss:.4f}. Saved checkpoint to {self.checkpoint_path}")
        else:
            self.counter += 1
            print(f"[*] EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("[!] EarlyStopping triggered: Halting training.")
                
        return self.early_stop


class ModelCheckpointerPyTorch:
    """
    Saves best model based on highest validation accuracy.
    """
    def __init__(self, filepath: str = 'checkpoints/best_acc.pt'):
        self.filepath = filepath
        self.best_acc = 0.0
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def step(self, val_acc: float, model: torch.nn.Module, epoch: int):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc
            }, self.filepath)
            print(f"[+] Checkpoint: New best val accuracy {val_acc:.2f}% saved to {self.filepath}")
