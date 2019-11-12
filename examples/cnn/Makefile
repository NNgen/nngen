TARGET=$(shell ls *.py | grep -v test | grep -v parsetab.py)
ARGS=

PYTHON=python3
#PYTHON=python
#OPT=-m pdb
#OPT=-m cProfile -s time
#OPT=-m cProfile -o profile.rslt

.PHONY: all
all: test

.PHONY: run
run:
	$(PYTHON) $(OPT) $(TARGET) $(ARGS)

.PHONY: test
test:
	$(PYTHON) -m pytest -vv

.PHONY: check
check:
	$(PYTHON) $(OPT) $(TARGET) $(ARGS) > tmp.v
	iverilog -tnull -Wall tmp.v
	rm -f tmp.v

.PHONY: clean
clean:
	rm -rf *.pyc __pycache__ parsetab.py .cache *.out *.png *.dot tmp.v uut.vcd 
