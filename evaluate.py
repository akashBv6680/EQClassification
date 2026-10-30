"""
CIFAR-10 Model Evaluation & Error Diagnostics
Implements:
  1. Loss & accuracy evaluation on test set
  2. Confusion matrix computation
  3. Extraction and visualization of sample misclassified images
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from models.baseline_cnn import BaselineCNNPyTorch
from models.resnet_cifar import ResNetCIFARPyTorch
from utils.dataset import get_pytorch_dataloaders, CIFAR10_MEAN, CIFAR10_STD

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_pytorch_dataloaders(batch_size=100)

    # Initialize model
    if args.model == 'resnet':
        model = ResNetCIFARPyTorch(num_classes=10)
    else:
        model = BaselineCNNPyTorch(num_classes=10)

    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"[*] Loaded weights from {args.checkpoint}")
    else:
        print(f"[!] Warning: Checkpoint {args.checkpoint} not found. Running with initialized weights.")

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []
    misclassified_samples = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Collect misclassified examples
            mis_mask = preds.ne(labels)
            if mis_mask.any() and len(misclassified_samples) < 20:
                for idx in torch.where(mis_mask)[0]:
                    img_np = images[idx].cpu().numpy().transpose(1, 2, 0)
                    # Denormalize image
                    img_np = img_np * CIFAR10_STD + CIFAR10_MEAN
                    img_np = np.clip(img_np, 0.0, 1.0)

                    misclassified_samples.append({
                        'image': img_np,
                        'true_class': CLASSES[labels[idx].item()],
                        'pred_class': CLASSES[preds[idx].item()],
                        'confidence': probs[idx][preds[idx]].item() * 100.0
                    })
                    if len(misclassified_samples) >= 20:
                        break

    accuracy = np.mean(np.array(all_preds) == np.array(all_targets)) * 100.0
    print(f"\n================ EVALUATION SUMMARY ================")
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    print(f"Total Evaluated Test Samples: {len(all_targets)}")
    print(f"\nClassification Report:\n")
    print(classification_report(all_targets, all_preds, target_names=CLASSES, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix shape:", cm.shape)

    if args.save_misclassified and len(misclassified_samples) > 0:
        os.makedirs('reports', exist_ok=True)
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i, ax in enumerate(axes.flat):
            if i < len(misclassified_samples):
                sample = misclassified_samples[i]
                ax.imshow(sample['image'])
                ax.set_title(f"True: {sample['true_class']}\nPred: {sample['pred_class']} ({sample['confidence']:.1f}%)", color='red')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig('reports/misclassified_samples.png', dpi=200)
        print("[+] Saved misclassified samples figure to reports/misclassified_samples.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_resnet.pt')
    parser.add_argument('--model', type=str, default='resnet', choices=['baseline', 'resnet'])
    parser.add_argument('--save-misclassified', action='store_true')
    args = parser.parse_args()
    evaluate(args)
