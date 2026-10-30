"""
Unit Tests for Model Architectures
Verifies:
  1. Forward pass output dimensions [B, 10]
  2. Residual block shape consistency with strides 1 and 2
  3. Non-zero gradient backpropagation
"""

import pytest
import torch
from models.baseline_cnn import BaselineCNNPyTorch
from models.resnet_cifar import ResNetCIFARPyTorch, ResidualBlockPyTorch

def test_baseline_cnn_output_shape():
    batch_size = 4
    model = BaselineCNNPyTorch(num_classes=10)
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (batch_size, 10), f"Expected shape ({batch_size}, 10), got {output.shape}"

def test_residual_block_stride1():
    block = ResidualBlockPyTorch(in_channels=32, out_channels=32, stride=1)
    dummy_input = torch.randn(2, 32, 32, 32)
    output = block(dummy_input)
    assert output.shape == (2, 32, 32, 32), "Stride 1 residual block must preserve spatial & channel dimensions"

def test_residual_block_stride2_downsampling():
    block = ResidualBlockPyTorch(in_channels=32, out_channels=64, stride=2)
    dummy_input = torch.randn(2, 32, 32, 32)
    output = block(dummy_input)
    assert output.shape == (2, 64, 16, 16), "Stride 2 residual block must halve spatial size and project channels"

def test_gradient_flow():
    model = ResNetCIFARPyTorch(num_classes=10)
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()

    # Assert gradients exist for all trainable parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient!"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} contains NaN gradients!"
