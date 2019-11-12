PYTHON=python3
#PYTHON=python
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
	find . -maxdepth 1 -type d | grep "./" | xargs -I {} make clean -C {}
