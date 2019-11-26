from setuptools import setup, find_packages

import os


def read(filename):
    # return open(os.path.join(os.path.dirname(__file__), filename), encoding='utf8').read()
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


setup(name='nngen',
      version=read('nngen/VERSION').splitlines()[0],
      description='A Fully-Customizable Hardware Synthesis Compiler for Deep Neural Network',
      long_description=read('README.md'),
      long_description_content_type="text/markdown",
      keywords='Neural Network, Deep Learning, FPGA, High-Level Synthesis',
      author='Shinya Takamaeda-Yamazaki',
      license="Apache License 2.0",
      url='https://github.com/NNgen/nngen',
      packages=find_packages(),
      package_data={'nngen': ['VERSION'], },
      install_requires=['Jinja2>=2.10',
                        'pyverilog>=1.2.0',
                        'veriloggen>=1.8.0',
                        'numpy>=1.14',
                        'onnx>=1.6.0'],
      extras_require={
          'test': ['pytest>=3.2', 'pytest-pythonpath>=0.7'],
      },
      )
