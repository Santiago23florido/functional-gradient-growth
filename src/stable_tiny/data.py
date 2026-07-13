"""Synthetic datasets used by the initial GroMo baseline."""

from __future__ import annotations

import gzip
import os
import pickle
import struct
import warnings
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset


class SyntheticDataLoader(torch.utils.data.DataLoader):
    """Minimal deterministic synthetic loader.

    It mirrors the loader used in GroMo's tutorial: every iteration resets the
    seed, so repeated training epochs see the same synthetic batches.
    """

    def __init__(
        self,
        nb_sample: int = 1,
        batch_size: int = 100,
        seed: int = 0,
        device: torch.device | None = None,
        in_features: int = 1,
        out_features: int = 1,
    ) -> None:
        self.nb_sample = nb_sample
        self.batch_size = batch_size
        self.seed = seed
        self.sample_index = 0
        self.device = device or torch.device("cpu")
        self.in_features = in_features
        self.out_features = out_features

    def __iter__(self) -> SyntheticDataLoader:
        torch.manual_seed(self.seed)
        self.sample_index = 0
        return self

    def __next__(self) -> Any:
        if self.sample_index >= self.nb_sample:
            raise StopIteration
        self.sample_index += 1

    def __len__(self) -> int:
        return self.nb_sample


class MultiSinDataLoader(SyntheticDataLoader):
    r"""Generate ``y[d] = sum_i sin((i + 1) * x[i] + d)`` samples."""

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        super().__next__()
        x = torch.randn(self.batch_size, self.in_features, device=self.device)
        y = torch.empty(self.batch_size, self.out_features, device=self.device)

        for d in range(self.out_features):
            y[:, d] = sum(
                torch.sin((i + 1) * x[:, i] + d) for i in range(self.in_features)
            )

        return x, y


class SmoothSinDataLoader(SyntheticDataLoader):
    r"""Generate a smoother sinusoidal regression task.

    The target uses only the first ``active_features`` inputs and normalizes the
    result. It is smoother than ``MultiSinDataLoader`` but still includes
    moderate frequencies and pairwise interactions.
    """

    def __init__(
        self,
        nb_sample: int = 1,
        batch_size: int = 100,
        seed: int = 0,
        device: torch.device | None = None,
        in_features: int = 3,
        out_features: int = 1,
        active_features: int = 2,
        frequency: float = 1.0,
        phase_shift: float = 0.5,
        interaction_strength: float = 0.25,
        linear_strength: float = 0.1,
    ) -> None:
        super().__init__(
            nb_sample=nb_sample,
            batch_size=batch_size,
            seed=seed,
            device=device,
            in_features=in_features,
            out_features=out_features,
        )
        self.active_features = max(1, min(active_features, in_features))
        self.frequency = frequency
        self.phase_shift = phase_shift
        self.interaction_strength = interaction_strength
        self.linear_strength = linear_strength

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        super().__next__()
        x = torch.randn(self.batch_size, self.in_features, device=self.device)
        y = torch.empty(self.batch_size, self.out_features, device=self.device)

        active_x = x[:, : self.active_features]
        feature_weights = torch.linspace(
            1.0,
            2.0,
            self.active_features,
            device=self.device,
        )
        for d in range(self.out_features):
            sinusoidal = torch.sin(
                self.frequency * feature_weights * active_x + self.phase_shift * d
            ).mean(dim=1)
            interaction = torch.sin(active_x[:, 0] * active_x[:, -1] + d)
            linear = active_x.mean(dim=1)
            y[:, d] = (
                sinusoidal
                + self.interaction_strength * interaction
                + self.linear_strength * linear
            )

        return x, y


MultiSinDataloader = MultiSinDataLoader


def make_cifar10_dataloaders(
    *,
    data_dir: str | None = None,
    train_samples: int | None = 5_000,
    validation_samples: int | None = 1_000,
    test_samples: int | None = 1_000,
    batch_size: int = 64,
    grayscale: bool = True,
    seed: int = 0,
    num_classes: int = 10,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return CIFAR-10 loaders with one-hot targets for the MSE FGD pipeline."""
    if num_classes != 10:
        raise ValueError("CIFAR-10 requires data.out_features = 10.")

    candidate_roots: list[Path] = []
    if data_dir is not None:
        candidate_roots.append(Path(data_dir))
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate_roots.append(
            Path(conda_prefix) / "datasets" / "cifar-10-batches-py"
        )
    candidate_roots.extend(
        [
            Path.home()
            / ".keras"
            / "datasets"
            / "cifar-10-batches-py-target"
            / "cifar-10-batches-py",
            Path("data") / "cifar-10-batches-py",
        ]
    )
    root = next((candidate for candidate in candidate_roots if candidate.exists()), None)
    if root is None:
        searched = ", ".join(str(candidate) for candidate in candidate_roots)
        raise FileNotFoundError(
            "CIFAR-10 batches not found. Expected the standard "
            f"'cifar-10-batches-py' directory in one of: {searched}"
        )

    def load_batch(filename: str) -> tuple[torch.Tensor, torch.Tensor]:
        with (root / filename).open("rb") as handle:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"dtype\(\): align should be passed.*",
                )
                batch = pickle.load(handle, encoding="bytes")
        x = torch.from_numpy(batch[b"data"].copy()).float()
        y = torch.tensor(batch[b"labels"], dtype=torch.long)
        return x, y

    train_parts = [load_batch(f"data_batch_{index}") for index in range(1, 6)]
    train_x = torch.cat([part[0] for part in train_parts], dim=0)
    train_labels = torch.cat([part[1] for part in train_parts], dim=0)
    test_x, test_labels = load_batch("test_batch")

    if grayscale:
        train_x = train_x.view(-1, 3, 1024).mean(dim=1)
        test_x = test_x.view(-1, 3, 1024).mean(dim=1)

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    generator = torch.Generator().manual_seed(seed)
    train_order = torch.randperm(train_x.shape[0], generator=generator)
    if train_samples is None:
        train_samples = train_x.shape[0] - (validation_samples or 0)
    if validation_samples is None:
        validation_samples = train_x.shape[0] - train_samples
    requested_train = train_samples + validation_samples
    if requested_train > train_x.shape[0]:
        raise ValueError(
            "CIFAR-10 train_samples + validation_samples exceeds the "
            f"available train set ({requested_train} > {train_x.shape[0]})."
        )

    train_indices = train_order[:train_samples]
    validation_indices = train_order[train_samples:requested_train]
    test_order = torch.randperm(test_x.shape[0], generator=generator)
    if test_samples is None:
        test_samples = test_x.shape[0]
    if test_samples > test_x.shape[0]:
        raise ValueError(
            "CIFAR-10 test_samples exceeds the available test set "
            f"({test_samples} > {test_x.shape[0]})."
        )
    test_indices = test_order[:test_samples]

    def one_hot(labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

    loader_generator = torch.Generator().manual_seed(seed + 3)
    train_loader = DataLoader(
        TensorDataset(train_x[train_indices], one_hot(train_labels[train_indices])),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        TensorDataset(
            train_x[validation_indices],
            one_hot(train_labels[validation_indices]),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(test_x[test_indices], one_hot(test_labels[test_indices])),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, validation_loader, test_loader


def _mnist_root_candidates(data_dir: str | None) -> list[Path]:
    candidates: list[Path] = []
    if data_dir is not None:
        root = Path(data_dir)
        candidates.extend([root, root / "raw", root / "MNIST" / "raw"])
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.extend(
            [
                Path(conda_prefix) / "datasets" / "mnist" / "raw",
                Path(conda_prefix) / "datasets" / "MNIST" / "raw",
            ]
        )
    candidates.extend(
        [
            Path("data") / "mnist" / "raw",
            Path("data") / "MNIST" / "raw",
        ]
    )
    return candidates


def _read_mnist_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file magic in {path}: {magic}")
        raw = handle.read()
    images = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return images.reshape(count, rows * cols).float() / 255.0


def _read_mnist_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file magic in {path}: {magic}")
        raw = handle.read()
    labels = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    return labels.reshape(count)


def make_mnist_dataloaders(
    *,
    data_dir: str | None = None,
    train_samples: int | None = 10_000,
    validation_samples: int | None = 2_000,
    test_samples: int | None = 2_000,
    batch_size: int = 64,
    seed: int = 0,
    num_classes: int = 10,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return MNIST loaders with one-hot targets for the MSE FGD pipeline."""
    if num_classes != 10:
        raise ValueError("MNIST requires data.out_features = 10.")

    required_files = (
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    )
    candidates = _mnist_root_candidates(data_dir)
    root = next(
        (
            candidate
            for candidate in candidates
            if all((candidate / filename).exists() for filename in required_files)
        ),
        None,
    )
    if root is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            "MNIST IDX gzip files not found. Expected train/test image and "
            f"label files in one of: {searched}"
        )

    train_x = _read_mnist_images(root / "train-images-idx3-ubyte.gz")
    train_labels = _read_mnist_labels(root / "train-labels-idx1-ubyte.gz")
    test_x = _read_mnist_images(root / "t10k-images-idx3-ubyte.gz")
    test_labels = _read_mnist_labels(root / "t10k-labels-idx1-ubyte.gz")

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    generator = torch.Generator().manual_seed(seed)
    train_order = torch.randperm(train_x.shape[0], generator=generator)
    if train_samples is None:
        train_samples = train_x.shape[0] - (validation_samples or 0)
    if validation_samples is None:
        validation_samples = train_x.shape[0] - train_samples
    requested_train = train_samples + validation_samples
    if requested_train > train_x.shape[0]:
        raise ValueError(
            "MNIST train_samples + validation_samples exceeds the available "
            f"train set ({requested_train} > {train_x.shape[0]})."
        )

    train_indices = train_order[:train_samples]
    validation_indices = train_order[train_samples:requested_train]
    test_order = torch.randperm(test_x.shape[0], generator=generator)
    if test_samples is None:
        test_samples = test_x.shape[0]
    if test_samples > test_x.shape[0]:
        raise ValueError(
            "MNIST test_samples exceeds the available test set "
            f"({test_samples} > {test_x.shape[0]})."
        )
    test_indices = test_order[:test_samples]

    def one_hot(labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

    loader_generator = torch.Generator().manual_seed(seed + 3)
    train_loader = DataLoader(
        TensorDataset(train_x[train_indices], one_hot(train_labels[train_indices])),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        TensorDataset(
            train_x[validation_indices],
            one_hot(train_labels[validation_indices]),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(test_x[test_indices], one_hot(test_labels[test_indices])),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, validation_loader, test_loader
