.PHONY: clean
clean:
	rm -rf *.pyc __pycache__ parsetab.py .cache *.out *.png *.dot tmp.v uut.vcd
	find . -maxdepth 1 -type d | grep "./" | xargs -I {} make clean -C {}
