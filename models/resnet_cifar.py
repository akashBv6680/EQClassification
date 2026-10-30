"""
Part 2 Option A: Residual Network Architecture for CIFAR-10
Dual Implementation: PyTorch & TensorFlow
"""

# ============================================================================
# PyTorch Implementation
# ============================================================================
import torch
import torch.nn as nn

class ResidualBlockPyTorch(nn.Module):
    """
    A simplified ResNet Block with identity skip connection:
    y = F(x) + x
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlockPyTorch, self).__init__()
        
        # Primary convolution branch F(x)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection (projection if spatial size or channels change)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection addition
        out = self.relu(out)
        return out


class ResNetCIFARPyTorch(nn.Module):
    """
    Mini-ResNet tailored specifically for 32x32 CIFAR-10 inputs.
    Contains:
      - Initial stem convolution (32 channels)
      - Stage 1: 2 ResBlocks (32 channels, 32x32)
      - Stage 2: 2 ResBlocks (64 channels, stride 2 -> 16x16)
      - Stage 3: 2 ResBlocks (128 channels, stride 2 -> 8x8)
      - Global Average Pooling + Linear(128, 10)
    """
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.2):
        super(ResNetCIFARPyTorch, self).__init__()
        
        # Stem
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual Stages
        self.stage1 = self._make_stage(32, 32, num_blocks=2, stride=1)
        self.stage2 = self._make_stage(32, 64, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(64, 128, num_blocks=2, stride=2)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128, num_classes)
        
        self._init_weights()

    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        curr_in = in_channels
        for s in strides:
            layers.append(ResidualBlockPyTorch(curr_in, out_channels, stride=s))
            curr_in = out_channels
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.gap(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


# ============================================================================
# TensorFlow / Keras Implementation
# ============================================================================
import tensorflow as tf
from tensorflow.keras import layers, Model

def residual_block_tf(x: tf.Tensor, filters: int, stride: int = 1) -> tf.Tensor:
    shortcut = x
    
    # Path F(x)
    out = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    
    out = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')(out)
    out = layers.BatchNormalization()(out)
    
    # Match shortcut projection if needed
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
    out = layers.add([out, shortcut])
    out = layers.ReLU()(out)
    return out

def create_resnet_cifar_tf(num_classes: int = 10, dropout_rate: float = 0.2) -> Model:
    inputs = layers.Input(shape=(32, 32, 3))
    
    # Stem
    x = layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Stage 1: 32 filters
    x = residual_block_tf(x, 32, stride=1)
    x = residual_block_tf(x, 32, stride=1)
    
    # Stage 2: 64 filters
    x = residual_block_tf(x, 64, stride=2)
    x = residual_block_tf(x, 64, stride=1)
    
    # Stage 3: 128 filters
    x = residual_block_tf(x, 128, stride=2)
    x = residual_block_tf(x, 128, stride=1)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation=None)(x)
    
    return Model(inputs=inputs, outputs=outputs, name='resnet_cifar_tf')
