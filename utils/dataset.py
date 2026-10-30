"""
CIFAR-10 Data Pipeline
Implements:
  1. Mean/Std Normalization [0.4914, 0.4822, 0.4465]
  2. Data Augmentation (RandomCrop 4px padding, RandomHorizontalFlip)
  3. High-throughput loaders (torch.utils.data.DataLoader with pin_memory and tf.data with AUTOTUNE prefetching)
"""

import os
from typing import Tuple

# ============================================================================
# PyTorch DataLoader Pipeline
# ============================================================================
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def get_pytorch_dataloaders(
    data_dir: str = './data',
    batch_size: int = 64,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates optimized PyTorch DataLoaders with data augmentation and normalization.
    """
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode='reflect'),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader


# ============================================================================
# TensorFlow tf.data Pipeline
# ============================================================================
import tensorflow as tf

def get_tf_datasets(
    batch_size: int = 64
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Creates high-throughput TensorFlow tf.data pipelines with prefetching & caching.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Cast to float32 and normalize
    mean = tf.constant(CIFAR10_MEAN, dtype=tf.float32)
    std = tf.constant(CIFAR10_STD, dtype=tf.float32)

    def preprocess_train(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        # Augmentations
        image = tf.image.resize_with_crop_or_pad(image, 40, 40)
        image = tf.image.random_crop(image, size=[32, 32, 3])
        image = tf.image.random_flip_left_right(image)
        # Normalize
        image = (image - mean) / std
        return image, tf.squeeze(label)

    def preprocess_test(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - mean) / std
        return image, tf.squeeze(label)

    # Train dataset pipeline
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = (train_ds
                .shuffle(buffer_size=10000, seed=42)
                .map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

    # Test dataset pipeline
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = (test_ds
               .map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(batch_size)
               .cache()
               .prefetch(tf.data.AUTOTUNE))

    return train_ds, test_ds
