PYTHON=python3
#PYTHON=python

.PHONY: all
all: clean

.PHONY: test
test:
	$(PYTHON) -m pytest -vv tests examples

.PHONY: clean
clean:
	make clean -C ./nngen
	make clean -C ./examples
	make clean -C ./tests
	rm -rf *.egg-info build dist *.pyc __pycache__ parsetab.py .cache *.out *.png *.dot tmp.v uut.vcd

.PHONY: release
release:
	pandoc README.md -t rst > README.rst
