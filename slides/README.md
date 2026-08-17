# CAGE-NAS presentation

This directory contains the English Beamer presentation for the CAGE-NAS
research project. The deck is intentionally modular: each section is stored in
`sections/` and included from `main.tex`.

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
