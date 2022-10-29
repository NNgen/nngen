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
      install_requires=['veriloggen>=2.1.1',
                        'numpy>=1.17',
                        'onnx>=1.9.0'],
      extras_require={
          'test': ['pytest>=3.8.1', 'pytest-pythonpath>=0.7.3'],
      },
      )
