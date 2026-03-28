SPHINXBUILD ?= sphinx-build
SOURCEDIR = .
BUILDDIR = _build

.PHONY: html doctest clean

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

doctest:
	$(SPHINXBUILD) -b doctest $(SOURCEDIR) $(BUILDDIR)/doctest

clean:
	rm -rf $(BUILDDIR)
