"""The theorem the grow-to-certify method rests on.

The method's claim is that growth can make Lemma 3.5's condition hold *by
construction*, in finitely many steps. That claim is a theorem only because
of three facts, and these tests pin all three rather than citing them:

1. Function-preserving growth leaves ``f`` **identical** yet strictly
   enlarges the tangent space ``T = range(J)``.
2. Therefore ``eps`` falls from growth **alone**, with no training step: the
   functional gradient ``r`` is unchanged, but ``P_T(r)`` moves closer to it.
3. ``eps < 1/2`` is exactly ``||P_T(r)||^2 > 0.8 ||r||^2`` -- the tangent
   space capturing more than 80 % of the gradient energy.

Together: with ``f`` fixed, every addition strictly increases
``||P_T(r)||``, so ``eps`` decreases monotonically and crosses 1/2 in
finitely many growths. If (1) were false the loop would be pointless; if (2)
were false it would not terminate.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import torch
from torch.func import functional_call, jacrev

from fgdlib.search.growth import grow_layer
from fgdlib.tangent import (
    _compute_exact_tangent_projection_step,
    tiny_optimal_update_kwargs,
)
from stable_tiny.pipeline import build_model, load_pipeline_config


def _fixture(hidden_size: int = 3, samples: int = 200):
    """A model whose tangent space is NOT already saturated.

    The rank of ``J`` is bounded by ``min(n_outputs, n_parameters)``. With few
    samples the Jacobian is already full row rank, growth cannot enlarge the
    tangent space at all, and the test would be vacuous -- so the probe must
    be much larger than the parameter count.
    """
    device = torch.device("cpu")
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        model=replace(
            config.model, hidden_size=hidden_size, number_hidden_layers=2
        ),
        fgd_approx=replace(config.fgd_approx, projection_solver="exact"),
    )
    model = build_model(config, device)
    torch.manual_seed(0)
    inputs = torch.randn(samples, config.data.in_features)
    targets = torch.randn(samples, config.data.out_features)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, targets), batch_size=50
    )
    return config, model, inputs, targets, loader, device


def _tangent_rank(model, inputs) -> int:
    """``rank(J)`` = the dimension of the reachable direction space."""
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    parameters = tuple(dict(model.named_parameters())[name] for name in names)
    buffers = OrderedDict(model.named_buffers())

    def outputs(values):
        state = OrderedDict(zip(names, values))
        state.update(buffers)
        return functional_call(model, state, (inputs,)).reshape(-1)

    jacobian = jacrev(outputs)(parameters)
    rows = jacobian[0].shape[0]
    matrix = torch.cat([part.reshape(rows, -1) for part in jacobian], dim=1)
    return int(torch.linalg.matrix_rank(matrix))


def _grow(config, model, loader, device, layer_index) -> None:
    grow_layer(
        model=model,
        train_loader=loader,
        layer_index=layer_index,
        device=device,
        line_search_config=config.scaling_line_search,
        optimal_update_kwargs=tiny_optimal_update_kwargs(
            config.fgd_approx, compute_delta=True
        ),
        function_preserving=True,
        preservation_tolerance=1e-3,
    )


def test_preserving_growth_keeps_f_but_enlarges_the_tangent_space() -> None:
    """Fact 1 -- the one that sounds contradictory and is not.

    ``f`` is a POINT; ``T`` is the set of DIRECTIONS it can move in. The new
    neuron carries outgoing weight ``omega = 0``, so it contributes nothing
    to ``f``; but ``df/domega != 0``, so moving it WOULD change ``f`` -- a
    genuinely new tangent direction.
    """
    config, model, inputs, _, loader, device = _fixture()
    model.eval()
    with torch.no_grad():
        before = model(inputs).clone()
    rank_before = _tangent_rank(model, inputs)

    _grow(config, model, loader, device, layer_index=0)

    model.eval()
    with torch.no_grad():
        after = model(inputs)
    # f is unchanged (float32 round-off only) ...
    assert torch.allclose(before, after, atol=1e-5)
    # ... while the space of reachable directions strictly grew.
    assert _tangent_rank(model, inputs) > rank_before


def test_epsilon_falls_from_growth_alone_without_any_training_step() -> None:
    """Fact 2 -- what makes the loop terminate.

    ``eps = ||P_T(r) - r|| / ||P_T(r)||`` depends on BOTH ``r`` and ``T``.
    Preserving growth fixes ``r`` and enlarges ``T``, so the projection moves
    closer to ``r`` and eps falls -- before any step is taken.
    """
    config, model, inputs, targets, loader, device = _fixture()

    def epsilon() -> float:
        step = _compute_exact_tangent_projection_step(
            model=model, x=inputs, y=targets, config=config.fgd_approx
        )
        return step.output_error.relative_error

    trajectory = [epsilon()]
    for index in range(4):
        _grow(config, model, loader, device, layer_index=index % 2)
        trajectory.append(epsilon())

    # Strictly decreasing, with no training step anywhere in between.
    assert all(
        later < earlier
        for earlier, later in zip(trajectory, trajectory[1:])
    ), trajectory


def test_the_condition_is_eighty_percent_of_the_gradient_energy() -> None:
    """Fact 3 -- what eps < 1/2 actually asks for.

    From the bridge identity ``||r||^2 = ||g||^2 (1 + eps^2)``:
        eps < 1/2  <=>  ||g||^2 > ||r||^2 / 1.25 = 0.8 ||r||^2
    """
    config, model, inputs, targets, _, _ = _fixture()
    step = _compute_exact_tangent_projection_step(
        model=model, x=inputs, y=targets, config=config.fgd_approx
    )
    error = step.output_error
    captured = error.approximation_norm**2          # ||g||^2
    total = error.target_norm**2                    # ||r||^2
    epsilon = error.relative_error

    # The identity itself, on real measured quantities. The tolerance is
    # float32 through a 600x57 least-squares solve, not the identity's own
    # accuracy: test_senn_expansion_score pins it to 1e-9 in float64.
    assert abs(total - captured * (1 + epsilon**2)) < 1e-2 * max(total, 1.0)
    # And the equivalence the method is built on.
    assert (epsilon < 0.5) == (captured > 0.8 * total)
