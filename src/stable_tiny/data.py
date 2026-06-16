"""Data loaders for the growth experiments.

Two task families are provided:

- ``teacher`` (default): a synthetic classification task where labels are
  produced by a *fixed random teacher MLP*. It needs zero extra dependencies,
  lives natively on the GPU, is instant to build, and -- crucially -- it is
  capacity-demanding, so adding neurons (growth) actually matters. That makes it
  a clean probe for the post-growth loss spike phenomenon.

- ``mnist`` / ``fashion_mnist``: standard image classification, loaded lazily
  via ``torchvision`` (only imported if requested). If torchvision is not
  installed a clear error is raised telling you how to enable it.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- #
# Synthetic "random teacher" classification task
# --------------------------------------------------------------------------- #
def _make_teacher(
    in_features: int,
    out_features: int,
    hidden: int,
    generator: torch.Generator,
) -> torch.nn.Module:
    """A fixed random 2-hidden-layer ReLU MLP used to generate labels."""
    teacher = torch.nn.Sequential(
        torch.nn.Linear(in_features, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, out_features),
    )
    # Deterministic init from the provided generator, then freeze.
    for p in teacher.parameters():
        if p.ndim >= 2:
            torch.nn.init.kaiming_normal_(p, generator=generator)
        else:
            p.data.normal_(0.0, 0.1, generator=generator)
        p.requires_grad_(False)
    return teacher.eval()


def _teacher_split(
    teacher: torch.nn.Module,
    n_samples: int,
    in_features: int,
    label_noise: float,
    generator: torch.Generator,
) -> TensorDataset:
    x = torch.randn(n_samples, in_features, generator=generator)
    with torch.no_grad():
        logits = teacher(x)
    y = logits.argmax(dim=1)
    if label_noise > 0:
        flip = torch.rand(n_samples, generator=generator) < label_noise
        n_flip = int(flip.sum())
        if n_flip:
            y = y.clone()
            y[flip] = torch.randint(
                0, logits.shape[1], (n_flip,), generator=generator
            )
    return TensorDataset(x, y)


def make_teacher_dataloaders(
    *,
    in_features: int = 20,
    out_features: int = 10,
    teacher_hidden: int = 128,
    n_train: int = 20_000,
    n_test: int = 4_000,
    label_noise: float = 0.0,
    batch_size: int = 256,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, dict]:
    """Build train/test loaders for the random-teacher classification task."""
    teacher_gen = torch.Generator().manual_seed(seed)
    teacher = _make_teacher(in_features, out_features, teacher_hidden, teacher_gen)

    train_gen = torch.Generator().manual_seed(seed + 1)
    test_gen = torch.Generator().manual_seed(seed + 2)
    train_ds = _teacher_split(teacher, n_train, in_features, label_noise, train_gen)
    test_ds = _teacher_split(teacher, n_test, in_features, 0.0, test_gen)

    loader_gen = torch.Generator().manual_seed(seed + 3)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=loader_gen
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    meta = {"in_features": in_features, "out_features": out_features}
    return train_loader, test_loader, meta


# --------------------------------------------------------------------------- #
# Gaussian-mixture ("blobs") classification — a known-good, *realizable* task
# --------------------------------------------------------------------------- #
def make_blobs_dataloaders(
    *,
    in_features: int = 16,
    out_features: int = 6,
    cluster_std: float = 1.6,
    center_scale: float = 6.0,
    n_train: int = 6000,
    n_test: int = 1500,
    batch_size: int = 64,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, dict]:
    """Build train/test loaders for a Gaussian-mixture classification task.

    Each class is an isotropic Gaussian blob with a fixed random center. Unlike
    the random-teacher task (whose labels come from a large unrealizable network),
    this task has a clear, achievable ceiling: a modest MLP separates the blobs to
    high accuracy, so growth that adds usable capacity visibly helps. The class
    overlap (``cluster_std`` vs ``center_scale``) sets the difficulty.
    """
    center_gen = torch.Generator().manual_seed(seed)
    centers = torch.randn(out_features, in_features, generator=center_gen) * center_scale

    def _split(n: int, gen: torch.Generator) -> TensorDataset:
        labels = torch.randint(0, out_features, (n,), generator=gen)
        x = centers[labels] + cluster_std * torch.randn(
            n, in_features, generator=gen
        )
        return TensorDataset(x, labels)

    train_ds = _split(n_train, torch.Generator().manual_seed(seed + 1))
    test_ds = _split(n_test, torch.Generator().manual_seed(seed + 2))

    loader_gen = torch.Generator().manual_seed(seed + 3)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=loader_gen
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    meta = {"in_features": in_features, "out_features": out_features}
    return train_loader, test_loader, meta


# --------------------------------------------------------------------------- #
# Optional torchvision image tasks
# --------------------------------------------------------------------------- #
def make_torchvision_dataloaders(
    *,
    name: str = "mnist",
    data_dir: str = "./data",
    n_train: int | None = None,
    batch_size: int = 256,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, dict]:
    """Build train/test loaders for MNIST / FashionMNIST (lazy torchvision)."""
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "torchvision is not installed in this environment. Either install it "
            "(careful: match it to torch 2.12) or use task='teacher' (the default, "
            "no extra deps)."
        ) from exc

    cls = {"mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST}[name]
    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = cls(data_dir, train=True, download=True, transform=tf)
    test_ds = cls(data_dir, train=False, download=True, transform=tf)

    if n_train is not None and n_train < len(train_ds):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(train_ds), generator=g)[:n_train]
        train_ds = torch.utils.data.Subset(train_ds, idx.tolist())

    loader_gen = torch.Generator().manual_seed(seed + 3)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=loader_gen
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    meta = {"in_features": (1, 28, 28), "out_features": 10}
    return train_loader, test_loader, meta


def get_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, dict]:
    """Dispatch on ``cfg['task']``."""
    task = cfg.get("task", "teacher")
    if task == "blobs":
        return make_blobs_dataloaders(
            in_features=cfg["in_features"],
            out_features=cfg["out_features"],
            cluster_std=cfg.get("cluster_std", 1.6),
            center_scale=cfg.get("center_scale", 6.0),
            n_train=cfg["n_train"],
            n_test=cfg["n_test"],
            batch_size=cfg["batch_size"],
            seed=cfg["seed"],
        )
    if task == "teacher":
        return make_teacher_dataloaders(
            in_features=cfg["in_features"],
            out_features=cfg["out_features"],
            teacher_hidden=cfg["teacher_hidden"],
            n_train=cfg["n_train"],
            n_test=cfg["n_test"],
            label_noise=cfg["label_noise"],
            batch_size=cfg["batch_size"],
            seed=cfg["seed"],
        )
    if task in ("mnist", "fashion_mnist"):
        return make_torchvision_dataloaders(
            name=task,
            data_dir=cfg.get("data_dir", "./data"),
            n_train=cfg.get("n_train"),
            batch_size=cfg["batch_size"],
            seed=cfg["seed"],
        )
    raise ValueError(f"Unknown task: {task!r}")
