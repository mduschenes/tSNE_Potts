# -*- coding: utf-8 -*-
"""
Created on Fri May  4 23:29:37 2018

@author: Matt
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("*.pyx"),
    include_dirs=[numpy.get_include()]
)