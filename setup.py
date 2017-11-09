# -*- coding: utf-8 -*-
# setup.py
# author : Antoine Passemiers

import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("sim",   ["sim.pyx"], include_dirs = [np.get_include()]),
    Extension("utils", ["utils.py"], include_dirs = [np.get_include()]),
]

setup(
  name = "spatialgames",
  cmdclass = {"build_ext" : build_ext},
  ext_modules = ext_modules
)