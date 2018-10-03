#!/usr/bin/env python

from sys import version
if version < '2.2.3':
    from distutils.dist import DistributionMetadata
    DistributionMetadata.classifiers = None
    DistributionMetadata.download_url = None
try:
    from setuptools import setup, Extension
except ImportError as ie:
    raise ie('please install setuptools')
from Cython.Distutils import build_ext
import numpy


with open('README.md') as file:
    long_description = file.read()

setup(
    name = 'ViSAPy',
    version = '1.0',
    maintainer = 'Espen Hagen',
    maintainer_email = 'espehage@fys.uio.no',
    url = 'https://www.github.com/espenhgn/ViSAPy',
    packages = ['ViSAPy'],
    provides = ['ViSAPy'],
    description = 'ViSAPy (Virtual Spiking Activity in Python)',
    long_description = long_description,
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("ViSAPy.cyextensions", ["ViSAPy/cyextensions.pyx"],
                           include_dirs=[numpy.get_include()]),
                   Extension("ViSAPy.driftcell", ["ViSAPy/driftcell.pyx"],
                           include_dirs=[numpy.get_include()]),
                   ],
    classifiers=[
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Cython',
        'Operating System :: Linux :: Unix :: OSX',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Development Status :: 4 - Beta',
        ],
    requires = [
        'numpy', 'scipy', 'matplotlib', 'neuron', 'Cython', 'h5py', 'mpi4py', 'sqlite3',
        ]
)
