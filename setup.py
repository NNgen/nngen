from setuptools import setup, find_packages

import re
import os


def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


m = re.search(r'(\d+\.\d+(\.\d+)?(-.+)?)',
              read('nngen/VERSION').splitlines()[0])
version = m.group(1) if m is not None else '0.0.0'

setup(name='nngen',
      version=version,
      description='A Fully-Customizable Hardware Synthesis Compiler for Deep Neural Network',
      long_description=read('README.md'),
      long_description_content_type="text/markdown",
      keywords='Neural Network, Deep Learning, FPGA, High-Level Synthesis',
      author='Shinya Takamaeda-Yamazaki',
      license="Apache License 2.0",
      url='https://github.com/NNgen/nngen',
      packages=find_packages(),
      package_data={'veriloggen': ['VERSION'], },
      install_requires=['Jinja2>=2.10',
                        'pyverilog>=1.2.0',
                        'veriloggen>=1.8.0',
                        'numpy>=1.14',
                        'onnx>=1.6.0'],
      extras_require={
          'test': ['pytest>=3.2', 'pytest-pythonpath>=0.7'],
      },
      )
