import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import os
from torch.utils.data import Subset

def generate_imbalanced_data(dataset, num_classes, imbalance_type, imbalance_factor):
    """Generates an imbalanced version of a given dataset.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset.
        num_classes (int): The number of classes in the dataset.
        imbalance_type (str): The type of imbalance ('exp' or 'step').
        imbalance_factor (float): The factor controlling the degree of imbalance.

    Returns:
        torch.utils.data.Dataset: An imbalanced version of the dataset.
    """
    targets = np.asarray(dataset.targets)
    idx = np.arange(len(dataset))
    idx_by_class = [idx[targets == i] for i in range(num_classes)]

    img_max = len(dataset) / num_classes
    img_num_per_cls = []
    if imbalance_type == 'exp':
        for cls_idx in range(num_classes):
            num = int(img_max * (imbalance_factor**(cls_idx / (num_classes - 1.0))))
            img_num_per_cls.append(num)
    elif imbalance_type == 'step':
        for cls_idx in range(num_classes // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(num_classes // 2):
            img_num_per_cls.append(int(img_max * imbalance_factor))
    else:
        raise ValueError(f"Unknown imbalance type: {imbalance_type}")

    new_indices = []
    for class_idx, num_samples in enumerate(img_num_per_cls):
        if num_samples > len(idx_by_class[class_idx]):
            replace = True # In case the imbalance factor is very high for some reason
        else:
            replace = False
        sampled_indices = np.random.choice(idx_by_class[class_idx], num_samples, replace=replace)
        new_indices.extend(sampled_indices)

    return Subset(dataset, new_indices)

def mnist(data_dir = './data', imbalance_type=None, factor=0.1):
    """Downloads and returns MNIST dataset, optionally with applied imbalance.

    Args:
        data_dir (str): Directory to store the dataset.
        imbalance_type (str, optional): Type of imbalance ('exp' or 'step'). Defaults to None (no imbalance).
        factor (float, optional): Imbalance factor. Defaults to 0.1.

    Returns:
        tuple: (train_dataset, test_dataset, input_shape, num_classes)
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    input_shape = train_ds[0][0].shape

    if imbalance_type is not None:
        num_classes = 10
        train_ds = generate_imbalanced_data(train_ds, num_classes, imbalance_type, factor)

    return train_ds, test_ds, input_shape, 10

def cifar10(data_dir = './data', imbalance_type=None, factor=0.1):
    """Downloads and returns CIFAR-10 dataset, optionally with applied imbalance.

    Args:
        data_dir (str): Directory to store the dataset.
        imbalance_type (str, optional): Type of imbalance ('exp' or 'step'). Defaults to None (no imbalance).
        factor (float, optional): Imbalance factor. Defaults to 0.1.

    Returns:
        tuple: (train_dataset, test_dataset, input_shape, num_classes)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    input_shape = train_ds[0][0].shape

    if imbalance_type is not None:
        num_classes = 10
        train_ds = generate_imbalanced_data(train_ds, num_classes, imbalance_type, factor)

    return train_ds, test_ds, input_shape, 10

def get_datasets(task, data_dir = './data', imbalance_type=None, factor=0.1):
    """Returns the specified dataset, optionally with applied imbalance.

    Args:
        task (str): The name of the dataset ('mnist' or 'cifar10').
        data_dir (str): Directory to store the dataset.
        imbalance_type (str, optional): Type of imbalance ('exp' or 'step'). Defaults to None.
        factor (float, optional): Imbalance factor. Defaults to 0.1.

    Returns:
        tuple: (train_dataset, test_dataset, input_shape, num_classes)
    """
    if task == 'mnist':
        return mnist(data_dir, imbalance_type, factor)
    elif task == 'cifar10':
        return cifar10(data_dir, imbalance_type, factor)
    else:
        raise ValueError(f'Unknown task: {task}')

if __name__ == '__main__':
    # Example usage for imbalanced MNIST
    train_ds_imbalanced_mnist, test_ds_mnist, shape_mnist = mnist(imbalance_type='exp', factor=0.01)
    print(f"Imbalanced MNIST training set size: {len(train_ds_imbalanced_mnist)}")
    targets_mnist = np.asarray(train_ds_imbalanced_mnist.dataset.targets)[train_ds_imbalanced_mnist.indices]
    print("Number of samples per class in imbalanced MNIST:")
    for i in range(10):
        print(f"Class {i}: {np.sum(targets_mnist == i)}")

    # Example usage for imbalanced CIFAR-10
    train_ds_imbalanced_cifar, test_ds_cifar, shape_cifar = cifar10(imbalance_type='step', factor=0.2)
    print(f"\nImbalanced CIFAR-10 training set size: {len(train_ds_imbalanced_cifar)}")
    targets_cifar = np.asarray(train_ds_imbalanced_cifar.dataset.targets)[train_ds_imbalanced_cifar.indices]
    print("Number of samples per class in imbalanced CIFAR-10:")
    for i in range(10):
        print(f"Class {i}: {np.sum(targets_cifar == i)}")

    # Example using get_datasets
    train_ds_get, _, _ = get_datasets('cifar10', imbalance_type='exp', factor=0.05)
    print(f"\nImbalanced CIFAR-10 via get_datasets size: {len(train_ds_get)}")
    targets_get = np.asarray(train_ds_get.dataset.targets)[train_ds_get.indices]
    print("Number of samples per class in imbalanced CIFAR-10 (via get_datasets):")
    for i in range(10):
        print(f"Class {i}: {np.sum(targets_get == i)}")