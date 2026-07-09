"""Synthetic datasets used by the initial GroMo baseline."""

from __future__ import annotations

from typing import Any

import torch


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


MultiSinDataloader = MultiSinDataLoader
