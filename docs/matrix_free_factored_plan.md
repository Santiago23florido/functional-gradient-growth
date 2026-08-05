# Making the ladder consume the factorisation instead of `J`

## Where this stands

`configs/fgd/mftangent_ladder_N1024.yaml` already runs the ladder's own
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

| file | line | function | second function to add |
|---|---|---|---|
| `search/certify.py` | 223 | `exact_relative_error` | `factored_relative_error` |
| `search/damping.py` | 300 | `minimal_relative_error_from_system` | `..._from_factors` |
| `search/damping.py` | 363 | `select_projection_damping` | `select_projection_damping_factored` |
| `search/realize.py` | 316, 339, 346 | realisation path | factored variant |
| `search/exact_where.py` | 267, 401-403 | growth scoring | factored variant |

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
