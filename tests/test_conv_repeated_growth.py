"""A conv stack must survive growing more than once.

Both failures these tests pin are stale GroMo caches that a SINGLE growth
never reveals, so a smoke test that grows once passes while a real run dies
on the statistics pass after the second event.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgdlib.models.convstack import build_conv_stack_model
from fgdlib.search.depth import insert_identity_layer
from fgdlib.search.growth import grow_layer
from fgdlib.tangent import ScalingLineSearchConfig, compute_expressivity_bottlenecks
from stable_tiny.pipeline import load_pipeline_config

DEVICE = torch.device("cpu")

BASE_STACK = [
    {"conv": [2, 3]},
    {"conv": [2, 3]},
    "maxpool",
    {"conv": [2, 3]},
    {"conv": [2, 3]},
    "maxpool",
    {"avgpool": [3, 3]},
    "flatten",
    {"mlp": [2, 1]},
]

UPDATE_KWARGS = {
    "compute_delta": False,
    "use_covariance": True,
    "alpha_zero": False,
    "omega_zero": True,
    "use_projection": True,
    "ignore_singular_values": False,
    "use_fisher": False,
    "maximum_added_neurons": 1,
    "numerical_threshold": 1e-6,
    "statistical_threshold": 1e-3,
}


def model():
    torch.manual_seed(0)
    return build_conv_stack_model(
        BASE_STACK,
        in_features=784,
        out_features=10,
        device=DEVICE,
        input_shape=(1, 28, 28),
    )


def loader(n: int = 128, batch: int = 64):
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(n, 1, 28, 28, generator=generator)
    y = torch.zeros(n, 10)
    y[torch.arange(n), torch.randint(0, 10, (n,), generator=generator)] = 1.0
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=batch
    )


def fgd_config():
    return load_pipeline_config(
        "configs/fgd/family_ladder_matrix_free_N1024.yaml"
    ).fgd_approx


def test_six_growths_round_robin_stay_exactly_function_preserving() -> None:
    """The statistic-shape bug: it fires on the pass AFTER the first growth.

    GroMo's ``Conv2dGrowingModule.layer_in_extension`` re-pins S as
    ``(in_channels + use_bias) * k_h * k_w`` while its own ``compute_s_update``
    emits ``use_bias + in_channels * k_h * k_w`` -- 36 against 28 at three
    channels with a 3x3 kernel. LinearGrowingModule agrees with itself in both
    places, which is why the MLP path has never met this.
    """
    net = model()
    net.eval()
    x = next(iter(loader()))[0]
    with torch.no_grad():
        reference = net(x).clone()

    for step in range(6):
        grow_layer(
            model=net,
            train_loader=loader(),
            layer_index=step % 3,
            device=DEVICE,
            line_search_config=ScalingLineSearchConfig(),
            optimal_update_kwargs=UPDATE_KWARGS,
            function_preserving=True,
            preservation_tolerance=1e-5,
        )
        # The statistics pass is where a stale shape or a stale border helper
        # raises, so re-measuring after every growth is the actual assertion.
        bottlenecks = compute_expressivity_bottlenecks(
            net, loader(), DEVICE, fgd_config()
        )
        assert len(bottlenecks) == 3
        assert all(value >= 0.0 for value in bottlenecks)
        net.eval()
        with torch.no_grad():
            assert float((net(x) - reference).abs().max()) == 0.0

    assert [int(layer.in_neurons) for layer in net._growable_layers] == [4, 4, 4]


@pytest.mark.parametrize("position", [1, 2, 3, 4, 5])
def test_statistics_run_at_every_insertion_position(position: int) -> None:
    """The border-helper bug: inserting at 1 replaces conv B's predecessor.

    GroMo caches ``bordering_convolution`` sized to the predecessor's unfolded
    width and never invalidates it, so the next pass feeds a 19-channel tensor
    to a helper built for 10 and torch raises.
    """
    net = model()
    inserted = insert_identity_layer(net, position=position, device=DEVICE)
    where = next(
        (
            index
            for index, layer in enumerate(net._growable_layers)
            if layer is inserted
        ),
        None,
    )
    bottlenecks = compute_expressivity_bottlenecks(
        net, loader(), DEVICE, fgd_config()
    )
    assert len(bottlenecks) == len(net._growable_layers)
    if where is not None:
        assert bottlenecks[where] >= 0.0


def test_an_inserted_layer_can_then_be_grown_repeatedly() -> None:
    """Insertion then growth then measurement, which is the real sequence."""
    net = model()
    insert_identity_layer(net, position=2, device=DEVICE)
    for _ in range(3):
        grow_layer(
            model=net,
            train_loader=loader(),
            layer_index=1,  # the inserted layer
            device=DEVICE,
            line_search_config=ScalingLineSearchConfig(),
            optimal_update_kwargs=UPDATE_KWARGS,
            function_preserving=True,
            preservation_tolerance=1e-5,
        )
        compute_expressivity_bottlenecks(net, loader(), DEVICE, fgd_config())
    # Growing at the inserted location widens conv B, the width that was
    # pinned before the insertion.
    assert int(net.layers[1].out_channels) == 2 + 3


def test_the_measured_depth_bottlenecks_can_actually_arbitrate() -> None:
    """If every depth candidate scored 0.0 the criterion would be inert.

    MEASURED on the base stack: the five positions span four orders of
    magnitude (6.5e-06 to 5.9e-02), so a ranking between them is a
    measurement and not a tie-break.
    """
    net = model()
    scores = []
    for position in range(1, len(net.layers)):
        trial = copy.deepcopy(net)
        inserted = insert_identity_layer(trial, position=position, device=DEVICE)
        where = next(
            (
                index
                for index, layer in enumerate(trial._growable_layers)
                if layer is inserted
            ),
            None,
        )
        if where is None:
            continue
        bottlenecks = compute_expressivity_bottlenecks(
            trial, loader(), DEVICE, fgd_config()
        )
        scores.append(bottlenecks[where])

    assert len(scores) >= 4
    assert all(value > 0.0 for value in scores)
    assert max(scores) / min(scores) > 100.0
