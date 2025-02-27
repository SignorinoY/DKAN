from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.utils
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor

from src.config import Config, DataModuleConfig


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = DataModuleConfig.dataset_name,
        cache_dir: str | Path = Config.cache_dir,
        input_dim: int = DataModuleConfig.input_dim,
        num_labels: int = DataModuleConfig.num_classes,
        batch_size: int = DataModuleConfig.batch_size,
        train_size: float = DataModuleConfig.train_size,
        train_split: str = DataModuleConfig.train_split,
        test_split: str = DataModuleConfig.test_split,
        num_workers: int = DataModuleConfig.num_workers,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.train_size = train_size
        self.train_split = train_split
        self.test_split = test_split
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

        cache_dir_is_empty = len(os.listdir(self.cache_dir)) == 0

        if cache_dir_is_empty:
            rank_zero_info(f"[{str(datetime.now())}] Downloading dataset.")
            if self.dataset_name == "mnist":
                datasets.MNIST(root=self.cache_dir, train=True, download=True)
                datasets.MNIST(root=self.cache_dir, train=False, download=True)
            elif self.dataset_name == "fmnist":
                datasets.FashionMNIST(root=self.cache_dir, train=True, download=True)
                datasets.FashionMNIST(root=self.cache_dir, train=False, download=True)
            elif self.dataset_name == "kmnist":
                datasets.KMNIST(root=self.cache_dir, train=True, download=True)
                datasets.KMNIST(root=self.cache_dir, train=False, download=True)
        else:
            rank_zero_info(
                f"[{str(datetime.now())}] Data cache exists. Loading from cache."
            )

    def setup(self, stage: str | None = None) -> None:
        transform = Compose([
            ToTensor(), Normalize((0.5,), (0.5,))
        ])
        if stage == "fit" or stage is None:
            if self.dataset_name == "mnist":
                dataset = datasets.MNIST(root=self.cache_dir, train=True, download=True, transform=transform)
            elif self.dataset_name == "fmnist":
                dataset = datasets.FashionMNIST(root=self.cache_dir, train=True, download=True, transform=transform)
            elif self.dataset_name == "kmnist":
                dataset = datasets.KMNIST(root=self.cache_dir, train=True, download=True, transform=transform)
            n_train = len(dataset)
            self.train_data, self.val_data = torch.utils.data.random_split(
                dataset, [int(n_train * self.train_size), n_train - int(n_train * self.train_size)]
            )
            del dataset
        if stage == "test" or stage is None:
            if self.dataset_name == "mnist":
                self.test_data = datasets.MNIST(root=self.cache_dir, train=False, download=True, transform=transform)
            elif self.dataset_name == "fmnist":
                self.test_data = datasets.FashionMNIST(root=self.cache_dir, train=False, download=True, transform=transform)
            elif self.dataset_name == "kmnist":
                self.test_data = datasets.KMNIST(root=self.cache_dir, train=False, download=True, transform=transform)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

class SimulateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = DataModuleConfig.dataset_name,
        train_val_test_split: tuple[int, int, int] = (9000, 1000, 10000),
        batch_size: int = DataModuleConfig.batch_size,
        num_workers: int = DataModuleConfig.num_workers,
        seed: int = Config.seed
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

    def prepare_data(self) -> None:
        pl.seed_everything(self.seed)
        n_total = sum(self.train_val_test_split)
        if self.dataset_name == "type1":
            x = np.random.uniform(-1, 1, (n_total, 2))
        elif self.dataset_name == "type2":
            x = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_total)
        y = np.log(np.abs(x[:, 0]) + (x[:, 1] / 2) ** 2 + 1)
        y += np.random.normal(0, 0.1, n_total)
        self.data = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)


    def setup(self, stage: str | None = None) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = torch.utils.data.TensorDataset(*self.data)
            self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(
                dataset, self.train_val_test_split,
                generator=torch.Generator().manual_seed(42)
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.num_workers
        )
