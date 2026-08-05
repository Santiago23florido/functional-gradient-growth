"""Second copies of the ladder's spectral consumers, reading ``(W, V)``.

The originals in ``damping.py`` and ``certify.py`` are untouched and keep
serving ``family_order: [tangent]``. These are reached only when a system
carries ``factors``, which only the matrix-free family produces.

## Why a factorisation is enough

``J = W V^T`` with ``W = J V`` of shape ``(rows, k)`` and ``V`` of shape
``(P, k)`` with orthonormal columns. Take ``W = A S B^T``; then

    J = A S B^T V^T = A S (V B)^T

and ``V B`` has orthonormal columns because both factors do. So

    svd(J) = (A, S, V B)

EXACTLY, from an SVD of a ``(rows, k)`` matrix. Nothing here ever holds an
``(N K, P)`` object, let alone the ``O(P^2)`` Gram -- the one that costs ~40 GB
at MNIST width and is the reason this path exists.

Two consequences worth stating because they are what make the substitution
cheap rather than merely possible:

  - ``eps`` needs only ``A`` and ``S``. The filtered approximation is
    ``A diag(s^2/(s^2+lam)) A^T r``, which never mentions the right factor at
    all, so the parameter dimension does not enter.
  - the update needs the right factor only as ``V (B z)``: a ``(k, k)`` matmul
    followed by one ``(P, k)`` product. ``O(P k)``, linear in ``P``.

## What is approximated, and in which direction

The factorisation is exact for the ``J`` it was built from, but that ``J`` is
the Krylov reconstruction, which is rank-deficient by one: the process
converges to a ``(P-1)``-dimensional invariant subspace. MEASURED at P=25,
that puts ``eps`` 1.5e-03 LOW -- optimistic. Forcing the missing direction in
makes it three times worse, so the deficiency is real and not a tuning knob.
Certification of an actual step is measured on the displacement applied, never
on this number.
"""

from __future__ import annotations

import torch

from fgdlib.tangent import (
    ExactTangentSystem,
    FGDApproxConfig,
    _output_relative_error_from_tensors,
)

__all__ = [
    "factored_gram",
    "factored_minimal_relative_error",
    "factored_projection_solve",
    "factored_spectrum",
]


def factored_projection_solve(
    system: ExactTangentSystem,
    target: torch.Tensor,
    absolute_damping: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(J^T J + lam I) u = J^T t``, solved in the Krylov coordinates.

    ``J = W V^T`` with ``V`` orthonormal columns, so ``J^T J = V (W^T W) V^T``
    and the damped solution lies in ``span(V)``. Writing ``u = V c`` collapses
    the normal equations to

        (W^T W + lam I) c = W^T t

    a ``k x k`` system. The ``O(P^2)`` Gram is never built and the ``P x P``
    factorisation never happens -- at MNIST width that Gram alone is ~40 GB,
    which is the entire reason this path exists.

    Returns ``(u, J u)``, matching what ``_solve_tangent_projection`` returns,
    so the caller's arithmetic is unchanged.

    The restriction to ``span(V)`` is the approximation and it is real: the
    exact solution has components along the directions the Krylov process did
    not reach. MEASURED at P=641 with the damping the ladder picks, the two
    updates agree to cos 0.979 but differ by 20% in norm -- near-zero damping
    puts the weight on the smallest singular values, exactly where truncation
    bites. The step is re-certified on what it applies, so this costs
    efficiency rather than soundness.
    """
    if system.factors is None:
        return None
    left_factor, right_factor = system.factors
    left_factor = left_factor.to(dtype=torch.float64)
    right_factor = right_factor.to(dtype=torch.float64)
    flat_target = target.reshape(-1).to(dtype=torch.float64)
    if left_factor.shape[0] != flat_target.numel():
        return None

    columns = left_factor.shape[1]
    small_gram = left_factor.t() @ left_factor
    small_gram = small_gram + absolute_damping * torch.eye(
        columns, dtype=torch.float64
    )
    try:
        coefficients = torch.linalg.solve(
            small_gram, left_factor.t() @ flat_target
        )
    except RuntimeError:
        return None
    if not torch.isfinite(coefficients).all():
        return None
    return right_factor @ coefficients, left_factor @ coefficients


def factored_gram(system: ExactTangentSystem) -> torch.Tensor | None:
    """``J^T J = V (W^T W) V^T``, at ``O(P^2 k)`` compute instead of ``O(NK P^2)``.

    Still an ``O(P^2)`` OBJECT, so this is a compute win and not a memory one.
    It is here because ``exact_where`` scores candidate structures against the
    Gram itself rather than against a solve, so the ``P x P`` form is genuinely
    needed there; the other consumers avoid it entirely via
    ``factored_projection_solve``.
    """
    if system.factors is None:
        return None
    left_factor, right_factor = system.factors
    left_factor = left_factor.to(dtype=torch.float64)
    right_factor = right_factor.to(dtype=torch.float64)
    return right_factor @ (left_factor.t() @ left_factor) @ right_factor.t()


def factored_spectrum(
    system: ExactTangentSystem,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """``(A, S, V)`` such that ``svd(J) = (A, S, V B)`` -- see module docstring.

    ``B`` is returned folded into ``V`` already, so callers get the right
    singular vectors directly and never rebuild ``J``.
    """
    if system.factors is None:
        return None
    left_factor, right_factor = system.factors
    left_factor = left_factor.to(dtype=torch.float64)
    right_factor = right_factor.to(dtype=torch.float64)
    if left_factor.numel() == 0 or right_factor.numel() == 0:
        return None
    left, singular_values, inner = torch.linalg.svd(
        left_factor, full_matrices=False
    )
    if singular_values.numel() == 0:
        return None
    # svd returns V^T, so the right vectors of W are inner.t(); folding gives
    # the right singular vectors of J as V @ inner.t(), shape (P, k).
    return left, singular_values, right_factor @ inner.t()


def factored_minimal_relative_error(
    system: ExactTangentSystem,
    config: FGDApproxConfig,
    minimum_relative_damping: float,
) -> float:
    """``minimal_relative_error_from_system``, computed from the factors.

    Same quantity, same least-damping point, same formula -- only the spectrum
    arrives factored. The parameter dimension never appears: the filtered
    approximation lives entirely in the left factor.
    """
    spectrum = factored_spectrum(system)
    if spectrum is None:
        return float("inf")
    left, singular_values, _right = spectrum
    target = system.target.reshape(-1).to(dtype=torch.float64)
    if target.numel() == 0:
        return float("inf")
    scale = float(singular_values.max()) ** 2
    if not scale > 0.0:
        return float("inf")

    coefficients = left.t() @ target
    absolute = minimum_relative_damping * scale
    denominator = singular_values.square() + absolute
    approximation = left @ (singular_values.square() / denominator * coefficients)
    stats = _output_relative_error_from_tensors(
        approximation=approximation.to(system.target.dtype),
        target=system.target.reshape(-1),
        eps=config.eps,
    )
    value = stats.output_error.relative_error
    if value is None or not float(value) == float(value):
        return float("inf")
    return float(value)
