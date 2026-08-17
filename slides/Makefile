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
	# Also remove artifacts left by an accidental in-place LaTeX build.
	$(LATEXMK) -C $(SOURCE)
	rm -f main.aux main.fdb_latexmk main.fls main.log main.nav main.out \
		main.pdf main.snm main.synctex.gz main.toc
