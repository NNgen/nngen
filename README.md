NNgen
==============================

A Fully-Customizable Hardware Synthesis Compiler for Deep Neural Network

Copyright 2017, Shinya Takamaeda-Yamazaki and Contributors


License
==============================

Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)


What's NNgen?
==============================

NNgen is an open-sourced compiler to synthesize a model-specific hardware accelerator for deep neural networks. NNgen generates a Verilog HDL source code and an IP-core package (IP-XACT) of a DNN accelerator from an input model definition.

Generated hardware is all-inclusive, which includes processing engine, on-chip memory, on-chip network, DMA controller, and control circuits. So the generated hardware does not require any additional controls from external circuit or CPU after the processing is started.

The backend of NNgen employes Veriloggen, an open-sourced mixed-paradigm high-level synthesis compiler in Python. So you can customize NNgen for new DNN algorithms and applications.


Contribute to NNgen
==============================

NNgen project always welcomes questions, bug reports, feature proposals, and pull requests on [GitHub](https://github.com/NNgen/nngen).

Community manager
--------------------

As a manager of this project, the community manager leads community management, and promote software development and diffusion. 

Committers
--------------------

Committers are individuals who are granted the write access to the project. In order to contribute as a committer, the approval of the community manager is required. The area of contribution can take all forms, including code contributions and code reviews, documents, education, and outreach. Committers are essential for a high quality and healthy project. The community actively looks for new committers from contributors.

Reviewers
--------------------

Reviewers are individuals who actively contributed to the project and are willing to participate in the code review of new contributions. We identify reviewers from active contributors. The committers should explicitly solicit reviews from reviewers. High-quality code reviews prevent technical debt for long-term and are crucial to the success of the project. A pull request to the project has to be reviewed by at least one reviewer in order to be merged.

for questions, bug reports, and feature proposals
--------------------

Please leave your comment on the [issue tracker](https://github.com/NNgen/nngen/issues) on GitHub.

for pull requests
--------------------

Please check "CONTRIBUTORS.md" for the contributors who provided pull requests.

NNgen uses **pytest** for the integration testing. **When you send a pull request, please include a testing example with pytest.** 
To write a testing code, please refer the existing testing examples in "tests" directory.

If the pull request code passes all the tests successfully and has no obvious problem, it will be merged to the *develop* branch by the committers.


Installation
==============================

Requirements
--------------------

- Python3: 3.6 or later
- Icarus Verilog: 10.1 or later

```
sudo apt install iverilog
```

- Jinja2: 2.10 or later
- Pyverilog: 1.1.4 or later
- Veriloggen: 1.7.3 or later
- NumPy: 1.14 or later
- ONNX: 1.6.0 or later

```
pip3 install jinja2 pyverilog veriloggen numpy onnx
```

Install
--------------------

Now you can install NNgen using setup.py script.

```
python3 setup.py install
```

Optional requirements for testing
--------------------

These are required for automatic testing of **tests** and **examples**.
We recommend to install these testing library to verify experimental features.

- pytest: 3.2 or later
- pytest-pythonpath: 0.7 or later
- PyTorch: 1.3.1 or later
- torchvision: 0.4.2 or later

```
pip3 install pytest pytest-pythonpath torch torchvision
```

For fast RTL simulation, we recommend to install Verilator.

- Verilator: 3.916 or later

```
sudo apt install verilator
```

Optional requirements for documentation
--------------------

If you want generate a document file from the source code, please install these dependen softwares.

- TeX Live: 2015 or later
- dvipng: 1.15 or later

```
sudo apt install texlive-science texlive-fonts-recommended texlive-fonts-extra dvipng
```

- Sphinx: 2.10 or later
- sphinx_rtd_theme : 0.4.3 or later

```
pip3 install sphinx sphinx_rtd_theme
```
    
Another installation way
--------------------
    
The current NNgen and Veriloggen are under the aggresive development.
Instead of the standard installation, you can download (or git clone) and install the latest version of NNgen, Veriloggen, and other libraries from GitHub.

### Download the latest NNgen, Veriloggen, and Pyverilog from GitHub

```
git clone https://github.com/NNgen/nngen.git
git clone https://github.com/PyHDI/veriloggen.git
git clone https://github.com/PyHDI/Pyverilog.git
```

### Create symbolic links to Veriloggen and the other libraries from NNgen

Instead of the actual installations, please create symbolic links to the dependent libraries.

```
cd nngen
ln -s ../veriloggen/veriloggen
ln -s ../Pyverilog/pyverilog
```

Docker
--------------------

Dockerfile is available. You can try NNgen on Docker without any installation on your host platform.

```
cd docker
sudo docker build -t user/nngen .
sudo docker run --name nngen -i -t user/nngen /bin/bash
cd nngen/examples/mlp/
make
```


Examples and testing
==============================

There are some exapmles in **examples** and various testing codes in **tests**.
The testing codes are actually good small examples suggesting how to represent a desired function.

To run the testing codes, please type the following commands.

```
cd tests
python3 -m pytest .
```

If you use Verilator instead of Icarus Verilog for RTL simulation, set "--sim" option.

```
python3 -m pytest --sim=verilator .
```


Getting started
==============================

Let's begin NNgen by an example. 

Represent a DNN model by NNgen operators
--------------------

```
```

Specify hardware configurations
--------------------


Convert NNgen operators to hardware objects and Verilog HDL
--------------------


Simulate a generated hardware by Veriloggen and Verilog simulator
--------------------


Related project
==============================

[Veriloggen](https://github.com/PyHDI/veriloggen)
- A library for constructing a Verilog HDL source code in Python

[Pyverilog](https://github.com/PyHDI/Pyverilog)
- Python-based Hardware Design Processing Toolkit for Verilog HDL
