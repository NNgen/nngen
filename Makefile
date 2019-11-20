PYTHON=python3
SIMTYPE=iverilog

.PHONY: all
all: clean

.PHONY: test
test:
	$(PYTHON) -m pytest -vv --sim $(SIMTYPE)

.PHONY: clean
clean:
	make clean -C ./nngen
	make clean -C ./examples
	make clean -C ./tests
	rm -rf *.egg-info build dist *.pyc __pycache__ parsetab.py .cache *.out *.png *.dot tmp.v uut.vcd
	rm -rf *_v1_0
	rm -rf *.onnx
	rm -rf hello_nngen.v hello_nngen.out hello_nngen_v1_0 hello_nngen.npy

.PHONY: release
release:
	pandoc README.md -t rst > README.rst
