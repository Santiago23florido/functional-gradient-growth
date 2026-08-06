# Making the ladder consume the factorisation instead of `J`

## Where this stands

`configs/fgd/family_ladder_matrix_free_N1024.yaml` runs the ladder's own
algorithm with `J` built matrix-free (`_matrix_free_tangent_system` in
`tangent.py`, gated on `family_order == ("matrix_free_tangent",)`). It removes
the `O(P^2)` Gram -- the ~40 GB blocker at MNIST width -- but still materialises
`J`, which is `O(N K P)`. That is the remaining obstacle.

## The identity that makes it work

The Golub-Kahan run gives `J = W V^T` with

- `W = J V`, shape `(rows, k)`
- `V`, shape `(P, k)`, orthonormal COLUMNS

Take the SVD of the small factor, `W = A S B^T`. Then

    J = A S B^T V^T = A S (V B)^T

and `V B` has orthonormal columns because both factors do. So

    svd(J) = (A, S, V B)

**exactly**, computed from an SVD of a `(rows, k)` matrix. Memory is
`O(N K k + P k)` -- linear in `P`, never `O(N K P)` and never `O(P^2)`.

Every consumer below is SVD-based, which is why this substitution reaches all
of them.

## Rule for the edit

A SECOND function per changed one. The originals stay byte-identical and keep
serving `family_order: [tangent]`; the new ones are reached only from the gate.
`configs/fgd/family_ladder_N1024.yaml` must not change.

Carry the factors on `ExactTangentSystem` as a new optional field defaulting to
`None`, so the exact path constructs exactly what it constructs today.

## Sites (all of them)

ALL DONE. Each original is untouched and still serves `family_order: [tangent]`.

| file | second function | verified against the dense route |
|---|---|---|
| `search/certify.py` | gate -> `factored_minimal_relative_error` | singular values 5.1e-08, eps 1.7e-04 |
| `search/damping.py` | `select_projection_damping_factored` | same lambda to 1e-9, eps 1.7e-04 |
| `search/realize.py` | gate -> `factored_projection_solve` | cos(u) 1.000000, same residual to 1e-6 |
| `search/exact_where.py` | gate -> `factored_gram` | Gram gap 1.0e-15 |

## What each one costs now

| | before | after |
|---|---|---|
| eps | SVD of (NK, P) | SVD of (NK, k) |
| damping selection | same | same |
| realisation solve | P x P Gram + factorisation | **k x k solve** |
| growth scoring | O(NK P^2) to build the Gram | O(P^2 k), still a P x P object |

`exact_where` is the one that did not fully collapse: it scores candidate
structures by comparing Grams rather than by solving, so the dense `P x P`
form is genuinely needed there. That is a compute win, not a memory one, and
it is the remaining ceiling at MNIST width.

## The gap that is real, and is not precision

`select_projection_damping_factored` agrees on eps and on the selected lambda
but NOT on the update: MEASURED at P=641, cos 0.979 with a 20% norm gap. The
ladder picks a relative damping around 1e-12, and at near-zero damping the
solution is dominated by the smallest singular values -- exactly the
directions a rank-k subspace is blind to. `factored_projection_solve` at a
realistic damping (1e-6 relative) matches to cos 1.000000, so the sensitivity
is to the damping, not to the construction.

## What must be measured, not assumed

- `eps` from the factored route against the exact one on the ladder's probe.
  Reference already taken at P=25: exact 0.87767164, matrix-free J
  0.87773799.
- The lambda each route selects. Same lambda, or the comparison is void.
- Peak memory, which is the entire point. `O(N K k + P k)` vs `O(N K P)`.

## The bias to keep guarding

While `k < rank(J)` the retained subspace cannot see the directions outside
it, so `eps` looks BETTER than it is -- MEASURED 0.8104 against 0.8779 at
P=25, k=24. `k` is driven to the numerical rank and capped at `P`. Both
guards are load-bearing; neither is a tuning knob.
