# CAGE-NAS presentation

This directory contains the English Beamer presentation for the CAGE-NAS
research project. The deck is intentionally modular: each section is stored in
`sections/` and included from `main.tex`.

The bibliography in `sections/99_references.tex` reproduces all 17 references
from `docs/internship-santiago/neurips_2026.tex` in the parent StageFrugal
workspace.

## Build

From this directory, run:

```bash
make
```

The compiled presentation is written to `build/main.pdf`.

## Institutional assets

- `assets/ensta.png`: RGB horizontal ENSTA/IP Paris logo from the official
  [ENSTA visual identity page](https://www.ensta.fr/logo-et-charte-graphique).
- `assets/lisn.png`: full-color LISN logo from the official
  [LISN communication page](https://www.lisn.upsaclay.fr/pole-communication/).

Both images were whitespace-trimmed for layout purposes; the logo artwork was
not modified.

`assets/demeter_cosine_similarity.png` is an unchanged copy of Figure 4 from
the internship report (`docs/internship-santiago/figures/cosinesim.png`).
The `assets/cage_*.png` plots are unchanged copies of the CAGE-NAS synthetic
and MNIST result figures from the same report.

## Sections

- `sections/00_title.tex`: project title page.
- `sections/10_demeter_alignment.tex`: concise explanation of DEMETER's
  parametric/function-space misalignment, with Figure 4 from the report.
- `sections/20_adaptive_representations.tex`: summary of Section 2.4,
  including the approximate functional-gradient update and its descent and
  convergence conditions.
- `sections/30_cage_nas_overview.tex`: certified CAGE-NAS train-and-grow
  control flow and architecture-search rule.
- `sections/31_damped_tangent_family.tex`: regularized tangent projection,
  spectral filtering, and relative-error certification.
- `sections/32_nonlinear_displacement_family.tex`: finite curved trajectory,
  angular certification, and ray-optimal scaling for the nonlinear family.
- `sections/33_where_to_grow.tex`: counterfactual certificate comparison and
  expressivity-bottleneck fallback for selecting the growth location.
- `sections/40_results_controlled_benchmark.tex`: controlled synthetic
  train-and-grow behavior and final seed results.
- `sections/41_results_search_quality.tex`: percentile comparison against the
  fixed-architecture search space.
- `sections/42_results_fixed_retraining.tex`: paired comparison with retraining
  the discovered fixed architectures.
- `sections/43_results_scalability.tex`: MNIST behavior and the Jacobian
  memory/computation bottleneck.
- `sections/90_conclusions.tex`: final conclusions, contributions, and research
  perspectives from the internship report.
- `sections/99_references.tex`: all references from the internship report.
