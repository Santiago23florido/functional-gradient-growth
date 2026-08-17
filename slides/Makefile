LATEXMK := latexmk
OUTDIR := build
SOURCE := main.tex
PDF := $(OUTDIR)/main.pdf

.PHONY: all clean

all: $(PDF)

$(PDF): $(SOURCE) sections/00_title.tex assets/ensta.png assets/lisn.png
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(OUTDIR) $(SOURCE)

clean:
	$(LATEXMK) -C -outdir=$(OUTDIR) $(SOURCE)
