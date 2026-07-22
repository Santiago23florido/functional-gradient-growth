# FGD configs

## The certified growth-and-search method (current)
- `search_ce_unified.yaml` — the working method on MNIST: grows from 3x2 by the
  unified width+depth criterion (SENN expansion score + rank ceiling), with
  generalised R1 (`growth_lookahead_adequacy`). No budget, no schedule.
- `search_cifar_unified.yaml` — the same method on CIFAR-10 (grayscale).
- `search_cifar_regularized.yaml` — the same method on **colour** CIFAR-10 with
  regularization inside the growing MLP: dropout (eval-transparent) and
  per-feature batch-norm (function-preservingly grown in sync).
- `search_ce_uniform.yaml` — the uniform-widening baseline (kept: referenced by
  the depth-insertion test).

## Base / reference (used by the unit tests)
- `default.yaml` — the smooth-sin base config used across the unit tests.
- `mnist_3x2_all_families.yaml`, `mnist_3x2_fp_growth.yaml` — pinned by the
  family-ladder and function-preserving-growth tests.

## The RKHS method (separate approach)
- `rkhs_default.yaml`, `rkhs_mnist.yaml`, `rkhs_grow_mnist.yaml`.

## Verification (Phase 0, transient)
- `mnist_lookahead_{on,off}.yaml`, `cifar_lookahead_on.yaml` — the on/off parity
  check for generalised R1. Removable once that lands on main.
