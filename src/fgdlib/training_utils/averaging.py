"""Polyak-Ruppert averaging in FUNCTION space, to damp trajectory variance.

Motivation is a measurement, not a preference. Two *where* criteria produced
the byte-identical architecture (784->8->8->10, 6552 parameters) and still
differed by 1.55 points of test accuracy (90.25 % against 88.70 %). Only
0.15 of that was selection bias from reporting the best test epoch; the rest
is where the certified trajectory happened to land. At that spread no
single-run comparison can support a claim, so the variance has to be
attacked, not merely reported.

The framework already licenses the fix. `report/CROSS_ENTROPY_FGD.md` 3.1
proves ``L`` is **convex in f** -- for sum-MSE the Hessian is ``2 Id``, for
softmax cross-entropy it is ``diag(p) - p p^T``, a covariance and therefore
PSD. Convexity gives Jensen's inequality directly: for certified iterates
``f_1 ... f_k`` and their function-space average ``f_bar = (1/k) sum f_t``,

    L(f_bar) <= (1/k) sum_t L(f_t).                                     (*)

So the averaged *function* is at least as good as the average point of the
trajectory -- a theorem here, not a heuristic, and it is exactly the
property convexity was established for. Averaging the network *weights*
would carry no such guarantee, because ``L`` is convex in ``f`` and not in
the parameters; the average must be taken over outputs.

Two practical points:

* The average is only meaningful across iterates of the SAME architecture,
  so the accumulator must be reset whenever the structure grows.
* No model snapshots are stored. The running mean is kept over the logits
  the evaluation pass already computes, so the cost is one tensor per split.
"""

from __future__ import annotations

import torch

__all__ = ["FunctionSpaceAverage"]


class FunctionSpaceAverage:
    """Running mean of a model's outputs over certified iterates.

    Usage per split (validation, test): call :meth:`update` once per epoch
    with that epoch's logits, then read :attr:`mean` for the averaged
    function's predictions. :meth:`reset` on every growth event.
    """

    def __init__(self) -> None:
        self._sum: torch.Tensor | None = None
        self._count: int = 0

    def reset(self) -> None:
        """Forget the trajectory. Required whenever the architecture changes.

        Outputs of different architectures are still vectors of the same
        shape, so averaging across a growth would silently succeed while
        mixing two different function classes; (*) says nothing about that.
        """
        self._sum = None
        self._count = 0

    def update(self, outputs: torch.Tensor) -> None:
        """Accumulate one certified iterate's outputs."""
        detached = outputs.detach()
        if self._sum is None or self._sum.shape != detached.shape:
            self._sum = torch.zeros_like(detached, dtype=torch.float64)
            self._count = 0
        self._sum += detached.to(dtype=torch.float64)
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> torch.Tensor | None:
        """The averaged function's outputs, or ``None`` before any update."""
        if self._sum is None or self._count == 0:
            return None
        return self._sum / self._count

    def accuracy(self, targets: torch.Tensor) -> float | None:
        """Argmax accuracy of the averaged function.

        ``targets`` may be one-hot or class indices.
        """
        mean = self.mean
        if mean is None:
            return None
        labels = targets if targets.ndim == 1 else targets.argmax(dim=-1)
        predicted = mean.argmax(dim=-1).to(labels.device)
        return float((predicted == labels).to(dtype=torch.float64).mean())
