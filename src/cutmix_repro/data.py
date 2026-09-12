"""CIFAR-100 data loaders with the paper's normalization."""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

MEAN = tuple(x / 255.0 for x in (125.3, 123.0, 113.9))
STD = tuple(x / 255.0 for x in (63.0, 62.1, 66.7))


def _subset(dataset: Dataset, count: int | None) -> Dataset:
    if count is None or count <= 0 or count >= len(dataset):
        return dataset
    return Subset(dataset, list(range(count)))


def make_loaders(
    data_root: str,
    train_batch_size: int,
    val_batch_size: int,
    workers: int,
    *,
    download: bool,
    train_samples: int | None = None,
    val_samples: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize(MEAN, STD)
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
    )
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_set = _subset(datasets.CIFAR100(data_root, train=True, download=download, transform=train_transform), train_samples)
    val_set = _subset(datasets.CIFAR100(data_root, train=False, download=download, transform=val_transform), val_samples)
    common = {"num_workers": workers, "pin_memory": True, "persistent_workers": workers > 0}
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader

