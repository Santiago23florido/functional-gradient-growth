"""Topographic visualization of functional gradient descent in function space.

The empirical *function space* of the paper is the output (logit) space on a
fixed probe set: a snapshot of the network is the vector ``f = logits(probe)`` in
``R^{N*C}``.  Crucially this space has the *same dimension regardless of network
width*, so the descent trajectories of a small net and a grown net live in one
common space and can be compared directly.

This module:

1. loads the recorded function-space trajectories of one or more runs,
2. projects them to 2D with PCA over the *combined* set of snapshots,
3. draws the genuine functional loss ``L(f) = CE(softmax(f), y_probe)`` as a
   topographic (filled-contour) map -- this is exact because ``L`` depends only
   on ``f``, not on the parametrization, and
4. animates each descent as a path moving across the map, marking growth events.

The result is a literal picture of functional gradient descent: the landscape is
fixed, and growth is the trajectory gaining access to directions the smaller
tangent space could not reach.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402


_METHOD_LABELS = {
    "gromo_tiny": "Scheduled TINY + AdamW",
    "functional_certified_tiny": "Certificate-triggered growth",
    "functional_triggered_tiny": "Functional certificate + TINY (v1)",
    "baseline_mlp": "MLP",
}
_METHOD_COLORS = {
    "gromo_tiny": "tab:blue",
    "functional_certified_tiny": "tab:purple",
    "functional_triggered_tiny": "tab:orange",
    "baseline_mlp": "tab:green",
}


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean cross-entropy of ``logits`` (..., N, C) against ``labels`` (N,)."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_partition = np.log(np.exp(shifted).sum(axis=-1))
    index = np.broadcast_to(labels, shifted.shape[:-1])  # (..., N)
    correct = np.take_along_axis(shifted, index[..., None], axis=-1).squeeze(-1)
    return (log_partition - correct).mean(axis=-1)


class _Trajectory:
    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=True)
        self.logits = data["logits"].astype(np.float64)  # (T, N, C)
        self.labels = data["probe_labels"].astype(np.int64)  # (N,)
        self.params = data["params"]
        self.phase = data["phase"].astype(str)
        self.method = str(data["method"])
        self.flat = self.logits.reshape(self.logits.shape[0], -1)  # (T, N*C)
        self.loss = _cross_entropy(self.logits, self.labels)  # (T,)
        self.label = _METHOD_LABELS.get(self.method, self.method)
        self.color = _METHOD_COLORS.get(self.method, "tab:red")
        self.coords: np.ndarray | None = None  # filled by render


def _project_and_surface(trajectories, labels, n, c, grid, pad):
    """PCA the snapshots to 2D, fill ``t.coords``, and compute the exact loss grid."""
    stacked = np.concatenate([t.flat for t in trajectories], axis=0)
    mean = stacked.mean(axis=0)
    centered = stacked - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]  # (2, N*C)
    for t in trajectories:
        t.coords = (t.flat - mean) @ components.T  # (T, 2)

    all_coords = np.concatenate([t.coords for t in trajectories], axis=0)
    a_min, a_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    b_min, b_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    a_pad, b_pad = pad * (a_max - a_min + 1e-9), pad * (b_max - b_min + 1e-9)
    a_axis = np.linspace(a_min - a_pad, a_max + a_pad, grid)
    b_axis = np.linspace(b_min - b_pad, b_max + b_pad, grid)
    grid_a, grid_b = np.meshgrid(a_axis, b_axis)

    # Each (a, b) reconstructs a logit configuration f = mean + a*pc1 + b*pc2.
    recon = (
        mean[None, None, :]
        + grid_a[..., None] * components[0][None, None, :]
        + grid_b[..., None] * components[1][None, None, :]
    ).reshape(grid, grid, n, c)
    surface = _cross_entropy(recon, labels)  # (grid, grid)
    return grid_a, grid_b, surface


def render_landscape(
    trajectory_paths: list[str | Path],
    out_path: str | Path,
    *,
    grid: int = 70,
    pad: float = 0.18,
    fps: int = 8,
    static_only: bool = False,
) -> Path:
    """Render the function-space loss landscape with animated descent paths."""
    trajectories = [_Trajectory(Path(p)) for p in trajectory_paths]
    labels = trajectories[0].labels
    n, c = trajectories[0].logits.shape[1:]

    grid_a, grid_b, surface = _project_and_surface(trajectories, labels, n, c, grid, pad)

    fig, ax = plt.subplots(figsize=(9, 7))
    contour = ax.contourf(grid_a, grid_b, surface, levels=30, cmap="terrain")
    ax.contour(grid_a, grid_b, surface, levels=14, colors="k", alpha=0.25, linewidths=0.5)
    fig.colorbar(contour, ax=ax, label="functional loss  L(f) = CE(softmax(f), y)")
    ax.set_xlabel("PC 1 of probe logits (function space)")
    ax.set_ylabel("PC 2 of probe logits (function space)")
    ax.set_title("Functional gradient descent on the loss landscape")

    artists = []
    for t in trajectories:
        (line,) = ax.plot([], [], "-", color=t.color, lw=2, alpha=0.9, label=t.label)
        (head,) = ax.plot([], [], "o", color=t.color, ms=9, mec="white", mew=1.2)
        growth = ax.scatter([], [], marker="*", s=0, color=t.color, edgecolors="white", zorder=5)
        artists.append((line, head, growth))
    ax.legend(loc="upper right", fontsize=9)

    max_t = max(t.coords.shape[0] for t in trajectories)

    def _draw(frame: int):
        for t, (line, head, growth) in zip(trajectories, artists):
            k = min(frame + 1, t.coords.shape[0])
            xs, ys = t.coords[:k, 0], t.coords[:k, 1]
            line.set_data(xs, ys)
            head.set_data([xs[-1]], [ys[-1]])
            grown = t.phase[:k] == "post_grow"
            if grown.any():
                growth.set_offsets(t.coords[:k][grown])
                growth.set_sizes(np.full(int(grown.sum()), 220))
        return [a for trio in artists for a in trio]

    if static_only:
        _draw(max_t - 1)
        out_path = Path(out_path).with_suffix(".png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return out_path

    anim = FuncAnimation(fig, _draw, frames=max_t, interval=1000 // fps, blit=False)
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".gif":
        out_path = out_path.with_suffix(".gif")
    anim.save(out_path, writer=PillowWriter(fps=fps))
    # Also drop a static PNG of the final state next to the GIF.
    _draw(max_t - 1)
    fig.savefig(out_path.with_suffix(".png"), dpi=130)
    plt.close(fig)
    return out_path


def _surface_height(grid_a, grid_b, surface, a, b):
    """Bilinear-free nearest lookup of the surface height at projected (a, b)."""
    ai = np.clip(np.searchsorted(grid_a[0], a) - 1, 0, surface.shape[1] - 1)
    bi = np.clip(np.searchsorted(grid_b[:, 0], b) - 1, 0, surface.shape[0] - 1)
    return surface[bi, ai]


def render_landscape_3d(
    trajectory_paths: list[str | Path],
    out_path: str | Path,
    *,
    grid: int = 70,
    pad: float = 0.18,
    fps: int = 8,
    static_only: bool = False,
    lift: float = 0.04,
) -> Path:
    """Render the function-space loss landscape as a 3D relief with descent paths.

    Same construction as :func:`render_landscape`, but the exact functional loss
    is drawn as a 3D surface (relief) and each descent rides on top of it at its
    measured loss height.  The animation advances the paths while slowly rotating
    the view so the relief and the valleys are easy to read.
    """
    trajectories = [_Trajectory(Path(p)) for p in trajectory_paths]
    labels = trajectories[0].labels
    n, c = trajectories[0].logits.shape[1:]
    grid_a, grid_b, surface = _project_and_surface(trajectories, labels, n, c, grid, pad)

    span = float(surface.max() - surface.min())
    lift_abs = lift * span

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        grid_a, grid_b, surface, cmap="terrain", alpha=0.82,
        linewidth=0, antialiased=True, rcount=grid, ccount=grid,
    )
    ax.contour(grid_a, grid_b, surface, levels=12, cmap="terrain", offset=surface.min(), alpha=0.6)
    ax.set_xlabel("PC 1 (function space)")
    ax.set_ylabel("PC 2 (function space)")
    ax.set_zlabel("functional loss L(f)")
    ax.set_title("Functional gradient descent — 3D relief of the loss landscape")

    # Each trajectory rides on the relief at its own measured loss height.
    for t in trajectories:
        t.z = t.loss + lift_abs  # (T,)

    artists = []
    for t in trajectories:
        (line,) = ax.plot([], [], [], "-", color=t.color, lw=2.2, label=t.label)
        (head,) = ax.plot([], [], [], "o", color=t.color, ms=8, mec="white", mew=1.0)
        (stars,) = ax.plot([], [], [], "*", color=t.color, ms=16, mec="white", mew=0.8, ls="none")
        artists.append((line, head, stars))
    ax.legend(loc="upper left", fontsize=9)

    max_t = max(t.coords.shape[0] for t in trajectories)

    def _draw(frame: int):
        for t, (line, head, stars) in zip(trajectories, artists):
            k = min(frame + 1, t.coords.shape[0])
            xs, ys, zs = t.coords[:k, 0], t.coords[:k, 1], t.z[:k]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
            head.set_data([xs[-1]], [ys[-1]])
            head.set_3d_properties([zs[-1]])
            grown = t.phase[:k] == "post_grow"
            if grown.any():
                stars.set_data(t.coords[:k, 0][grown], t.coords[:k, 1][grown])
                stars.set_3d_properties(t.z[:k][grown])
        ax.view_init(elev=38, azim=-60 + 0.7 * frame)
        return [a for trio in artists for a in trio]

    if static_only:
        _draw(max_t - 1)
        out_path = Path(out_path).with_suffix(".png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return out_path

    anim = FuncAnimation(fig, _draw, frames=max_t, interval=1000 // fps, blit=False)
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".gif":
        out_path = out_path.with_suffix(".gif")
    anim.save(out_path, writer=PillowWriter(fps=fps))
    _draw(max_t - 1)
    fig.savefig(out_path.with_suffix(".png"), dpi=130)
    plt.close(fig)
    return out_path
