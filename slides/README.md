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
