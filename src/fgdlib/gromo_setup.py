"""Import helpers for the local GroMo checkout."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def ensure_gromo_importable() -> Path | None:
    """Make ``gromo`` importable from an install or the sibling checkout.

    Returns the path added to ``sys.path``. If GroMo is already importable,
    returns ``None``.
    """
    if importlib.util.find_spec("gromo") is not None:
        return None

    repo_root = Path(__file__).resolve().parents[2]
    stage_frugal_root = repo_root.parent

    candidates: list[Path] = []
    if env_path := os.environ.get("GROMO_SRC"):
        configured = Path(env_path).expanduser().resolve()
        candidates.extend([configured, configured / "src"])

    candidates.append(stage_frugal_root / "gromo" / "src")

    for candidate in candidates:
        if (candidate / "gromo").is_dir():
            sys.path.insert(0, str(candidate))
            return candidate

    searched = "\n".join(f"- {path}" for path in candidates)
    raise ModuleNotFoundError(
        "Could not import 'gromo'. Install GroMo or set GROMO_SRC to the local "
        f"GroMo repository/source path. Searched:\n{searched}"
    )
