NNgen
=====

Neural Network Accelerator Generator

Copyright 2017, Shinya Takamaeda-Yamazaki and Contributors

License
=======

Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

What’s NNgen?
=============

Under construction.

Contribute to NNgen
===================

NNgen project always welcomes questions, bug reports, feature proposals,
and pull requests on `GitHub <https://github.com/PyHDI/nngen>`__.

Community manager
-----------------

As a manager of this project, the community manager leads community
management, and promote software development and diffusion.

Committers
----------

Committers are individuals who are granted the write access to the
project. In order to contribute as a committer, the approval of the
community manager is required. The area of contribution can take all
forms, including code contributions and code reviews, documents,
education, and outreach. Committers are essential for a high quality and
healthy project. The community actively looks for new committers from
contributors.

Reviewers
---------

Reviewers are individuals who actively contributed to the project and
are willing to participate in the code review of new contributions. We
identify reviewers from active contributors. The committers should
explicitly solicit reviews from reviewers. High-quality code reviews
prevent technical debt for long-term and are crucial to the success of
the project. A pull request to the project has to be reviewed by at
least one reviewer in order to be merged.

for questions, bug reports, and feature proposals
-------------------------------------------------

Please leave your comment on the `issue
tracker <https://github.com/PyHDI/nngen/issues>`__ on GitHub.

for pull requests
-----------------

Please check “CONTRIBUTORS.md” for the contributors who provided pull
requests.

NNgen uses **pytest** for the integration testing. **When you send a
pull request, please include a testing example with pytest.** To write a
testing code, please refer the existing testing examples in “tests”
directory.

If the pull request code passes all the tests successfully and has no
obvious problem, it will be merged to the *develop* branch by the
committers.

Installation
============

Requirements
------------

-  Python3: 3.6 or later

-  Icarus Verilog: 10.1 or later

-  TeX Live: 2015 or later

-  dvipng: 1.15 or later

::

   sudo apt install iverilog texlive-science texlive-fonts-recommended texlive-fonts-extra dvipng

-  Pyverilog: 1.1.4 or later
-  Veriloggen: 1.7.3 or later
-  Jinja2: 2.10 or later
-  NumPy: 1.14 or later
-  Sphinx: 2.10 or later
-  sphinx_rtd_theme: 0.4.3 or later

::

   pip3 install veriloggen pyverilog jinja2 numpy sphinx sphinx_rtd_theme

Install
-------

As the current NNgen and Veriloggen are under the aggresive development,
installation of them into your system platform is NOT recommended.
Instead, please download (or git clone) the latest version of NNgen,
Veriloggen and other libraries from GitHub.

Download the latest NNgen, Veriloggen and other libraries from GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   git clone https://github.com/kmsysgi/nngen.git
   git clone https://github.com/PyHDI/veriloggen.git
   git clone https://github.com/PyHDI/Pyverilog.git

Create symbolic links to Veriloggen and the other libraries from NNgen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of the actual installations, please create symbolic links to the
dependent libraries.

::

   cd nngen
   ln -s ../veriloggen/veriloggen
   ln -s ../Pyverilog/pyverilog

Getting Started
===============

Under construction.

Related Project
===============

`Veriloggen <https://github.com/PyHDI/veriloggen>`__ - A library for
constructing a Verilog HDL source code in Python

`Pyverilog <https://github.com/PyHDI/Pyverilog>`__ - Python-based
Hardware Design Processing Toolkit for Verilog HDL
