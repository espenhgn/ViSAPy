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
  name = 'ViSAPy',
  version = '0.1',
  maintainer = 'Espen Hagen',
  maintainer_email = 'ehagen@fz-juelich.de',
  url = 'http://www.fz-juelich.de/inm/inm-6',
  packages = ['ViSAPy'],
  provides = ['ViSAPy'],
  description = 'Tool for generation of biophysically realistic benchmark data for evaluation of spike-sorting algorithms',
  long_description = 'Tool for generation of biophysically realistic benchmark data for evaluation of spike-sorting algorithms',
  cmdclass = {'build_ext': build_ext},
  ext_modules = [Extension("ViSAPy.cyextensions", ["ViSAPy/cyextensions.pyx"],
                         include_dirs=[numpy.get_include()]),
                 Extension("ViSAPy.driftcell", ["ViSAPy/driftcell.pyx"],
                         include_dirs=[numpy.get_include()]),
                 ],
)
