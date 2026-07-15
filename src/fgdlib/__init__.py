"""fgdlib: certified functional gradient descent for growing networks.

The library owns everything that trains the network: the tangent-space
FGD approximation with validation certificates (``fgdlib.tangent``), the
certified RKHS method with exact global-optimality constants
(``fgdlib.rkhs``), the GroMo growth machinery (``fgdlib.growth``), and
the training/optimization utilities. Datasets, experiment pipelines,
logging and plotting are intentionally NOT part of the library; they
consume it.
"""

from __future__ import annotations

__version__ = "0.1.0"
