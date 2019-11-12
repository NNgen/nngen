from setuptools import setup, find_packages

import re
import os


def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


m = re.search(r'(\d+\.\d+\.\d+(-.+)?)',
              read('nngen/VERSION').splitlines()[0])
version = m.group(1) if m is not None else '0.0.0'

setup(name='nngen',
      version=version,
      description='Neural Network Accelerator Generator',
      long_description=read('README.rst'),
      keywords='Neural Network, Deep Learning, FPGA, High-Level Synthesis',
      author='Shinya Takamaeda-Yamazaki',
      license="Apache License 2.0",
      url='https://sites.google.com/site/shinyaty',
      packages=find_packages(),
      package_data={'veriloggen': ['VERSION'], },
      install_requires=['veriloggen>=1.7.2',
                        'pyverilog>=1.1.4',
                        'Jinja2>=2.10',
                        'numpy>=1.14'],
      extras_require={
          'test': ['pytest>=3.2', 'pytest-pythonpath>=0.7'],
      },
      )
