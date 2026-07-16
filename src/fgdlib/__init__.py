"""fgdlib: certified functional gradient descent for growing networks.

The library owns everything that trains the network:

- ``fgdlib.adaptive`` -- strict Algorithm-1 FGD over the complete finite
  empirical output space, with disposable proposal families and
  function-preserving representation growth.
- ``fgdlib.tangent``  -- tangent-space FGD approximation with validation
  certificates (relative error, learning-rate interval, sensor checks).
- ``fgdlib.rkhs``     -- certified RKHS FGD (arXiv:2606.16926) with exact
  constants and the global-optimality certificate for a fixed structure.
- ``fgdlib.growth``   -- GroMo growth machinery (optimal extensions and
  the scaling line search).
- ``fgdlib.growth_schedule``, ``fgdlib.lr_scheduler``, ``fgdlib.optim``,
  ``fgdlib.training`` -- schedules, optimizers and training utilities.

Datasets, experiment pipelines, logging and plotting are intentionally
NOT part of the library; they consume it (see ``stable_tiny.pipeline``).
"""

from __future__ import annotations

from fgdlib.adaptive import (
    AdaptiveFGDAttemptRecord,
    AdaptiveFGDCertificate,
    AdaptiveFGDConfig,
    AdaptiveFGDSearchResult,
    AdaptiveGrowthResult,
    certify_empirical_secant,
    certify_empirical_secant_models,
    empirical_functional_loss,
    empirical_inner_product,
    empirical_model_functional_loss,
    empirical_norm,
    grow_layer_function_preserving,
    search_adaptive_fgd_step,
)
from fgdlib.empirical_pl import (
    EmpiricalPLConfig,
    EmpiricalPLEpochResult,
    EmpiricalPLStepRecord,
    EmpiricalPLTrainer,
)
from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.growth import (
    GrowthResult,
    ScalingLineSearchConfig,
    grow_layer,
)
from fgdlib.growth_schedule import (
    GrowthScheduleConfig,
    layer_index_for_growth,
    should_grow,
)
from fgdlib.lr_scheduler import (
    LRSchedulerConfig,
    apply_learning_rate,
    learning_rate_for_epoch,
)
from fgdlib.optim import OptimizerConfig, build_optimizer, current_learning_rate
from fgdlib.rkhs import (
    FGDRKHSConfig,
    FGDRKHSEpochResult,
    FGDRKHSStepRecord,
    FGDRKHSTheory,
    FGDRKHSTrainer,
    FrozenAffineFeatureMap,
    FrozenMLPFeatureMap,
    KernelDictionaryModel,
    default_level_ladder,
    median_heuristic_gamma,
    theory_descent_coefficient,
    theory_learning_rate_upper_bound,
)
from fgdlib.tangent import (
    FGDApproxConfig,
    FGDValidationCertificate,
    SecantFGDConfig,
    evaluate_fgd_validation_certificate,
    evaluate_secant_validation_certificate,
    train_one_epoch_fgd_approx,
)
from fgdlib.training import (
    RegressionMetrics,
    count_parameters,
    evaluate_regression_metrics,
    train_one_epoch,
)

__all__ = [
    "__version__",
    "ensure_gromo_importable",
    # strict empirical adaptive FGD
    "AdaptiveFGDAttemptRecord",
    "AdaptiveFGDCertificate",
    "AdaptiveFGDConfig",
    "AdaptiveFGDSearchResult",
    "AdaptiveGrowthResult",
    "certify_empirical_secant",
    "certify_empirical_secant_models",
    "empirical_functional_loss",
    "empirical_inner_product",
    "empirical_model_functional_loss",
    "empirical_norm",
    "grow_layer_function_preserving",
    "search_adaptive_fgd_step",
    # empirical-PL certified full-weight training
    "EmpiricalPLConfig",
    "EmpiricalPLEpochResult",
    "EmpiricalPLStepRecord",
    "EmpiricalPLTrainer",
    # growth
    "GrowthResult",
    "ScalingLineSearchConfig",
    "grow_layer",
    "GrowthScheduleConfig",
    "layer_index_for_growth",
    "should_grow",
    # schedules / optim / training
    "LRSchedulerConfig",
    "apply_learning_rate",
    "learning_rate_for_epoch",
    "OptimizerConfig",
    "build_optimizer",
    "current_learning_rate",
    "RegressionMetrics",
    "count_parameters",
    "evaluate_regression_metrics",
    "train_one_epoch",
    # certified RKHS FGD
    "FGDRKHSConfig",
    "FGDRKHSEpochResult",
    "FGDRKHSStepRecord",
    "FGDRKHSTheory",
    "FGDRKHSTrainer",
    "FrozenAffineFeatureMap",
    "FrozenMLPFeatureMap",
    "KernelDictionaryModel",
    "default_level_ladder",
    "median_heuristic_gamma",
    "theory_descent_coefficient",
    "theory_learning_rate_upper_bound",
    # tangent-space FGD
    "FGDApproxConfig",
    "FGDValidationCertificate",
    "SecantFGDConfig",
    "evaluate_fgd_validation_certificate",
    "evaluate_secant_validation_certificate",
    "train_one_epoch_fgd_approx",
]

__version__ = "0.1.0"
