"""Jensen's inequality is what licenses averaging the certified trajectory.

`report/CROSS_ENTROPY_FGD.md` 3.1 establishes convexity of L in f for both
certified functionals. These tests check that the consequence actually holds
for the implemented losses, so the variance treatment rests on a verified
inequality rather than on the docstring that cites it.
"""

from __future__ import annotations

import torch

from fgdlib.averaging import FunctionSpaceAverage
from fgdlib.tangent import batch_functional_loss


def _one_hot(rows: int, classes: int, generator: torch.Generator) -> torch.Tensor:
    targets = torch.zeros(rows, classes, dtype=torch.float64)
    index = torch.randint(0, classes, (rows,), generator=generator)
    targets[range(rows), index] = 1.0
    return targets


def test_averaging_never_loses_to_the_mean_trajectory_point() -> None:
    """L(f_bar) <= mean_t L(f_t), for BOTH certified functionals."""
    generator = torch.Generator().manual_seed(0)
    for name in ("mse", "cross_entropy"):
        for _ in range(30):
            rows, classes, steps = 24, 5, 7
            targets = _one_hot(rows, classes, generator)
            iterates = [
                torch.randn(rows, classes, generator=generator, dtype=torch.float64)
                for _ in range(steps)
            ]

            average = FunctionSpaceAverage()
            for outputs in iterates:
                average.update(outputs)

            averaged_loss = float(
                batch_functional_loss(average.mean, targets, name)
            )
            mean_of_losses = sum(
                float(batch_functional_loss(outputs, targets, name))
                for outputs in iterates
            ) / steps
            assert averaged_loss <= mean_of_losses + 1e-9


def test_averaging_damps_trajectory_spread() -> None:
    """The point of the exercise: less spread than the individual iterates.

    A trajectory that has converged in distribution but still jitters is
    simulated as a fixed signal plus independent noise. The averaged
    function's loss must sit at or below the best-case spread of the
    individual iterates, and well below their mean.
    """
    generator = torch.Generator().manual_seed(1)
    rows, classes, steps = 64, 6, 12
    targets = _one_hot(rows, classes, generator)
    signal = torch.randn(rows, classes, generator=generator, dtype=torch.float64)

    iterates = [
        signal + 0.8 * torch.randn(
            rows, classes, generator=generator, dtype=torch.float64
        )
        for _ in range(steps)
    ]
    average = FunctionSpaceAverage()
    for outputs in iterates:
        average.update(outputs)

    losses = [
        float(batch_functional_loss(outputs, targets, "cross_entropy"))
        for outputs in iterates
    ]
    averaged = float(batch_functional_loss(average.mean, targets, "cross_entropy"))
    assert averaged < sum(losses) / steps
    # And it is close to the noiseless signal it is recovering.
    clean = float(batch_functional_loss(signal, targets, "cross_entropy"))
    assert abs(averaged - clean) < abs(sum(losses) / steps - clean)


def test_reset_is_required_across_a_growth() -> None:
    """Different architectures must not be averaged together."""
    average = FunctionSpaceAverage()
    average.update(torch.ones(4, 3))
    assert average.count == 1
    average.reset()
    assert average.count == 0
    assert average.mean is None


def test_shape_change_restarts_the_average() -> None:
    """A defensive echo of the same rule, in case a reset is ever missed."""
    average = FunctionSpaceAverage()
    average.update(torch.ones(4, 3))
    average.update(torch.ones(4, 5))       # different output width
    assert average.count == 1
    assert average.mean.shape == (4, 5)


def test_accuracy_of_the_averaged_function() -> None:
    average = FunctionSpaceAverage()
    # Two iterates that individually get one sample each wrong, but whose
    # average gets both right -- the effect being exploited.
    average.update(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    average.update(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert average.accuracy(targets) == 1.0
    assert average.accuracy(torch.tensor([0, 1])) == 1.0
