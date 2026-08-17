LATEXMK := latexmk
OUTDIR := build
SOURCE := main.tex
PDF := $(OUTDIR)/main.pdf
SECTIONS := $(wildcard sections/*.tex)
ASSETS := $(wildcard assets/*)

.PHONY: all clean

all: $(PDF)

$(PDF): $(SOURCE) $(SECTIONS) $(ASSETS)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(OUTDIR) $(SOURCE)

clean:
	$(LATEXMK) -C -outdir=$(OUTDIR) $(SOURCE)
