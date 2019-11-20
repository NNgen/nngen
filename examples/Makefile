PYTHON=python3
SIMTYPE=iverilog

.PHONY: all
all: clean

.PHONY: test
test:
	$(PYTHON) -m pytest -vv --sim $(SIMTYPE)

.PHONY: run
run:
	find . -maxdepth 1 -type d | grep "./" | xargs -I {} make run -C {}

.PHONY: clean
clean:
	rm -rf *.pyc __pycache__ parsetab.py .cache *.out *.png *.dot tmp.v uut.vcd
	rm -rf *_v1_0
	rm -rf *.onnx
	find . -maxdepth 1 -type d | grep "./" | xargs -I {} make clean -C {}
