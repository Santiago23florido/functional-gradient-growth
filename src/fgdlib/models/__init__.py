"""Model construction on top of GroMo containers.

- ``regularized_mlp`` -- compose GroMo's growth-aware regularizers (dropout,
  and soon normalization) into a growing MLP without adding any growth
  mechanism of this library's own. Certification-safe: the certificate runs
  in eval, where these are the identity / use fixed statistics.
"""
