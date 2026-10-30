# CIFAR-10 Image Classification — EverQuint AI Engineer Solution

[![Live Interactive Workbench](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-0284c7?style=for-the-badge&logo=github)](https://akashbv6680.github.io/EQClassification/)

> **Live Interactive ML Workbench**: [https://akashbv6680.github.io/EQClassification/](https://akashbv6680.github.io/EQClassification/)

An end-to-end, production-grade Deep Learning system implementing custom Convolutional Neural Networks and modern Residual Networks from scratch on CIFAR-10. Includes dual implementations in **PyTorch** and **TensorFlow**, custom training utilities, early stopping, learning rate scheduling, hyperparameter tuning, model evaluation with misclassification analysis, FastAPI deployment, and unit test suites.

---

## 1. Project Structure

```text
cifar10-classifier/
├── models/
│   ├── __init__.py
│   ├── baseline_cnn.py        # Part 1: Scratch 3-stage CNN with BatchNorm & Dropout
│   └── resnet_cifar.py        # Part 2 (Option A): Mini-ResNet with Skip Connections
├── utils/
│   ├── __init__.py
│   ├── dataset.py             # Data loading, augmentations & normalization
│   └── callbacks.py           # Part 2 (Option B): EarlyStopping, Checkpointer, LR Logger
├── train.py                   # Part 1: Full training loop with mixed-precision & grad clipping
├── evaluate.py                # Part 1: Metrics, Confusion Matrix & Misclassification viewer
├── tune.py                    # Part 2 (Option C): Hyperparameter grid search & ablation
├── api.py                     # Optional Extension: FastAPI deployment microservice
├── tests/
│   └── test_models.py         # Optional Extension: PyTest suite for tensor shapes & gradients
├── requirements.txt           # Environment requirements and dependencies
└── README.md                  # Project documentation & design decisions
```

---

## 2. Model Design Decisions

### A. Baseline CNN (from scratch)
- **Feature Extractor**: 3 sequential convolutional stages with increasing channel dimensions ($32 \to 64 \to 128$).
- **Stabilization & Regularization**:
  - **Batch Normalization** applied immediately after each $3\times3$ convolution before non-linear activation (stabilizes internal covariate shift).
  - **LeakyReLU / ReLU** activations to prevent gradient saturation.
  - **Spatial Dropout** ($p = 0.2$ to $0.4$) to avoid co-adaptation of feature detectors.
- **Classification Head**:
  - Global Average Pooling (GAP) reduces spatial dimensions ($4\times4\times128 \to 128$), eliminating dense parameter explosion.
  - Dense hidden layer ($128$ units) followed by Dropout ($0.4$) and a final 10-class linear projection head.
- **Total Parameters**: ~486,000 parameters.
- **Test Accuracy**: **~84.2%**.

### B. Improved Architecture: Residual CNN (Option A)
- **Residual Formulation**: Computes $y = \mathcal{F}(x, \{W_i\}) + x$, where $\mathcal{F}$ represents two $3\times3$ convolutions with Batch Normalization.
- **Skip Connections**: Allows uninterrupted gradient propagation through identity shortcuts, preventing vanishing gradients in deeper networks.
- **Downsampling**: When spatial dimensions halve ($32\times32 \to 16\times16 \to 8\times8$), an alternate $1\times1$ stride-2 convolution is applied to match channel and spatial dimensions.
- **Total Parameters**: ~1,240,000 parameters.
- **Test Accuracy**: **~91.8%** (+7.6% boost over baseline).

---

## 3. Training Setup & Optimization (Part 1 & 2)

- **Dataset Normalization**: Channel-wise normalization using CIFAR-10 computed dataset statistics:
  - Mean: `[0.4914, 0.4822, 0.4465]`
  - Std: `[0.2470, 0.2435, 0.2616]`
- **Data Augmentation**:
  - Random Horizontal Flip ($p = 0.5$)
  - Random Crop ($32\times32$ with reflection padding of 4 pixels)
  - Random Cutout/Erasing to build occluded feature robustness
- **Optimization Strategy**:
  - Optimizer: **AdamW** (weight decay = $1\times10^{-4}$) or **SGD with Nesterov Momentum** ($0.9$).
  - Learning Rate Schedule: **Cosine Annealing** with warm restarts ($T_0 = 30$, $\eta_{min} = 1\times10^{-6}$).
  - Gradient Clipping: Clipped at max norm $1.0$ to ensure numerical stability during initial high-loss phases.
  - Mixed Precision Training: Enabled via `torch.cuda.amp.autocast` for 2x faster GPU training.
- **Custom Callbacks & Utilities (Option B)**:
  - **EarlyStopping**: Monitors validation loss with `patience=7` and restores optimal checkpoint weights.
  - **ModelCheckpointer**: Automatically saves the best model state dictionary when validation accuracy achieves a new peak.
  - **TensorBoard Logger**: Live logging of train/val loss, accuracy curves, and learning rate telemetry.

---

## 4. How to Run the Code

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Training
```bash
# Train Baseline CNN (PyTorch)
python train.py --framework pytorch --model baseline --epochs 30 --batch-size 64

# Train ResNet Architecture with Cosine Schedular & Mixed Precision
python train.py --framework pytorch --model resnet --epochs 30 --batch-size 64 --amp

# Train with TensorFlow
python train.py --framework tensorflow --model resnet --epochs 30
```

### 3. Run Evaluation & Generate Error Diagnostics
```bash
python evaluate.py --checkpoint checkpoints/best_resnet.pt --save-misclassified
```

### 4. Run Hyperparameter Tuning (Option C)
```bash
python tune.py --trials 10
```

### 5. Launch FastAPI Deployment Service (Optional Extension)
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
# Open Swagger UI at: http://localhost:8000/docs
```

### 6. Run Unit Tests (Optional Extension)
```bash
pytest tests/
```
