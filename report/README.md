# Report: Certificate-Driven Functional-Gradient Growth

NeurIPS-structured technical report for the certified-growth method in
`stable-tiny`.

## Build

```bash
make            # -> functional_growth.pdf  (runs pdflatex twice)
```

The preamble is self-contained (uses the standard `article` class with a
NeurIPS-like layout) so it compiles with a plain TeX Live install. To use the
official NeurIPS style instead, drop `neurips_2024.sty` next to the `.tex` and
swap the marked preamble block for `\usepackage{neurips_2024}`.

## Regenerating the inputs

The tables and figures are produced from the repo, not hand-edited:

```bash
# tables/tolerance.tex and tables/dataset.tex  (mean +- std over seeds, GPU)
python ../ablate.py --seeds 0 1 2

# figures/landscape.png and figures/landscape_3d.png
python ../make_landscape.py            # 2-D contour
python ../make_landscape.py --3d       # 3-D relief
cp ../results/*_landscape*.png figures/
```

All experiment runs are on GPU (`ablate.py` asserts CUDA is available).
