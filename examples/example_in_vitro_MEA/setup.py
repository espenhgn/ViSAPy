#!/usr/bin/env python

from sys import version
if version < '2.2.3':
    from distutils.dist import DistributionMetadata
    DistributionMetadata.classifiers = None
    DistributionMetadata.download_url = None
    
    
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
  name = 'MoI',
  version = '1.0',
  maintainer = 'Espen Hagen',
  maintainer_email = 'ehagen@fz-juelich.de',
  url = 'http://www.fz-juelich.de/inm/inm-6',
  packages = ['MoI'],
  provides = ['MoI'],
  description = 'Method of Images implementation for predicting extracellular potentials',
  long_description = 'Method of Images implementation for predicting extracellular potentials',
  cmdclass = {'build_ext': build_ext},
  ext_modules = [Extension("MoI.cython_funcs", ["MoI/cython_funcs.pyx"],
                         include_dirs=[numpy.get_include()]),
                 ],
)

