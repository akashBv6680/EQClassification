"""
CIFAR-10 Baseline CNN Architecture (from scratch)
Dual Implementation: PyTorch & TensorFlow
"""

# ============================================================================
# PyTorch Implementation
# ============================================================================
import torch
import torch.nn as nn

class BaselineCNNPyTorch(nn.Module):
    """
    Custom 3-stage Convolutional Neural Network built from scratch.
    Complies with EverQuint Part 1 requirements:
      - Convolutional layers
      - BatchNorm
      - Dropout
      - Classification head
    """
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super(BaselineCNNPyTorch, self).__init__()
        
        # Stage 1: 32 channels, 32x32 -> 16x16
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        # Stage 2: 64 channels, 16x16 -> 8x8
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate * 0.75)
        )
        
        # Stage 3: 128 channels, 8x8 -> 4x4
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Global Average Pooling & Classification Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Kaiming normal initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.gap(out)
        out = self.classifier(out)
        return out


# ============================================================================
# TensorFlow / Keras Implementation
# ============================================================================
import tensorflow as tf
from tensorflow.keras import layers, Model

def create_baseline_cnn_tf(num_classes: int = 10, dropout_rate: float = 0.3) -> Model:
    """
    TensorFlow/Keras implementation of the Baseline CNN from scratch.
    """
    inputs = layers.Input(shape=(32, 32, 3))
    
    # Stage 1: 32 channels
    x = layers.Conv2D(32, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.SpatialDropout2D(dropout_rate * 0.5)(x)
    
    # Stage 2: 64 channels
    x = layers.Conv2D(64, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.SpatialDropout2D(dropout_rate * 0.75)(x)
    
    # Stage 3: 128 channels
    x = layers.Conv2D(128, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.SpatialDropout2D(dropout_rate)(x)
    
    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation=None)(x)
    
    return Model(inputs=inputs, outputs=outputs, name='baseline_cnn_tf')
