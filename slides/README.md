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

## Sections

- `sections/00_title.tex`: project title page.
- `sections/10_demeter_alignment.tex`: concise explanation of DEMETER's
  parametric/function-space misalignment, with Figure 4 from the report.
- `sections/20_adaptive_representations.tex`: summary of Section 2.4,
  including the approximate functional-gradient update and its descent and
  convergence conditions.
- `sections/30_cage_nas_overview.tex`: certified CAGE-NAS train-and-grow
  control flow and architecture-search rule.
- `sections/99_references.tex`: all references from the internship report.
