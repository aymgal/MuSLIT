#!/usr/bin/env python

import os
import sys
from setuptools import find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

REQUIRES = ['numpy']
PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(
    name='MuSLIT',
    version='0.0.1',
    description='Multi-wavelength Sparse Linear Inversion Tool',
    long_description=read('README.md'),
    author='aymgal',
    author_email='aymeric.galan@gmail.com',
    url='https://github.com/aymgal/MuSLIT',
    download_url='https://github.com/aymgal/MuSLIT.git',
    packages=find_packages(PACKAGE_PATH),
    install_requires=REQUIRES,
    license='MIT',
    keywords='gravlens',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
)
