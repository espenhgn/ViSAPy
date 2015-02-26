ViSAPy
======

ViSAPy (Virtual Spiking Activity in Python) is a tool for generation of
biophysically realistic benchmark data for evaluation of spike sorting
algorithms.

The tool is accompanied by the scientific publication:

Espen Hagen, Torbj¿rn V. Ness, Amir Khosrowshahi, Christina S¿rensen,
Marianne Fyhn, Torkel Hafting, Felix Franke and Gaute T. Einevoll.
"ViSAPy: A Python tool for biophysics-based generation of virtual spiking
activity for evaluation of spike-sorting algorithms", Journal of
Neuroscience Methods,
Available online 4 February 2015, ISSN 0165-0270,
http://dx.doi.org/10.1016/j.jneumeth.2015.01.029.
(http://www.sciencedirect.com/science/article/pii/S0165027015000369)


ViSAPy was developed in the Computational Neuroscience Group
(http://compneuro.umb.no), Department of Mathemathical Sciences and Technology
(http://www.nmbu.no/imt) at the Norwegian University of Life Sciences
(http://www.nmbu.no) and Institute of Neuroscience and Medicine (INM-6) and
Institute for Advanced Simulation (IAS-6), JŸlich Research Centre and JARA,
JŸlich, Germany (http://www.fz-juelich.de/inm/inm-6/EN).

This work was supported by the Research Council of Norway (eVita, NOTUR,
NevroNor), the International Neuroinformatics Coordinating Facility (INCF)
through the Norwegian and German Nodes, the European Union Seventh Framework
Programme (FP7/2007-2013) under grant agreement no. 604102 ("Human Brain
Project"), the Helmholtz Portfolio Supercomputing and Modeling for the Human
Brain (SMHB), the European Community through the ERC Advanced Grant 267351
"NeuroCMOS", and the National Institutes of Health (NIH) through
NIH grant R01EY019965. 

This scientific software is released under the GNU Public License
(see LICENSE file).


Dependencies
============

ViSAPy is a module implemented using Python (http://www.python.org) and
facilitates on other packages and modules developed for Python by an active
open source community. To get started with (scientific) Python, a nice resource
would be for example these pages: https://scipy-lectures.github.io. 


To make ViSAPy work and depending on what sort of Python distribution is used,
the following packages may have to be installed:

- numpy (http://www.numpy.org)
- scipy (http://www.scipy.org)
- matplotlib (http://matplotlib.org)
- mpi4py (http://mpi4py.scipy.org)
- h5py (http://www.h5py.org)
- Cython (http://cython.org)


Note that pre-built Python distributions such as Anaconda
(https://store.continuum.io/cshop/anaconda/) or Enthought Canopy
(https://www.enthought.com/products/canopy/) may come with such packages
preinstalled. If not, depending on operating system, such packages can usually
be installed easily using the operating system's or other package managers
(``apt-get``, ``synaptic``, ``macports``) or using the ``easy_install`` or
``pip`` command line
tools:                                                                      ::

    pip install <package name> --user

or:                                                                         ::

    sudo pip install <package name>


Further, ViSAPy plotting and example scripts may require the Python SpikeSort
package (http://spikesort.org). 
The NEST (http://www.nest-initiative.org) and NEURON (http://neuron.yale.edu)
simulation softwares must also be built with Python bindings (please refer to
their respective installation instructions, but see below).

Finally ViSAPy is built around LFPy (http://compneuro.umb.no/LFPy) for computing
extracellular potentials around multicompartment neuron models and uses NEURON
internally. Detailed information on getting LFPy (and also setting up NEURON) is
given on the page http://compneuro.umb.no/LFPy/information.html

ViSAPy has been developed and tested on OSX and Linux platforms running Python
version 2.7.x. As NEST only do not support Windows, ViSAPy will only run on
Posix based platforms (OSX, Linux, Unix).


Installation
============

After making sure that all prerequisites above are met, download the ViSAPy
source codes from GitHub using the terminal. Make sure that ``git`` version
control software is installed (http://git-scm.com):                         ::

    cd /where/to/put/files
    git clone https://github.com/espenhgn/ViSAPy.git ViSAPy

At this point it is now possible to install ViSAPy as any other Python package
locally in the user's Python environment issuing the commands:              ::

    cd ViSAPy
    python setup.py install --user

However, as the user may want to modify or contribute to ViSAPy one can build
extension modules inplace and link the /path/to/ViSAPy to the PYTHONPATH
environment variable:                                                       ::
    
    cd ViSAPy
    python setup.py build_ext -i


Then, on linux, unix or OSX operating systems edit the .bashrc or .profile file
located in your home folder to include the line:                            ::

    export PYTHONPATH=/path/to/ViSAPy/:$PYTHONPATH


Documentation
=============

To generate the html documentation issue from the ViSAPy source code directory,
it is possible to do so using Sphinx (http://sphinx-doc.org) with the command:
::
    
    cd /path/to/ViSAPy
    sphinx-build -b html docs path/to/dest

The main html file with the autogenerated documentation of ViSAPy should now be
found in the location:                                                      ::
    
    path/to/dest/index.html


Examples
========

As example files for ViSAPy we have provided a host of scripts reproducing
results of our publication (http://dx.doi.org/10.1016/j.jneumeth.2015.01.029.).
The example files are found in:                                             ::
    
    /path/to/ViSAPy/examples
    
As an overview of the different example files:

- ``examples/ISI_waveforms/ISI_waveforms.py`` corresponds to Results section 3.1.2 and figure references therein
- ``examples/example_figure_02.py`` corresponds to Figure 2 in the publication
- ``examples/example_in_vivo_tetrode.py`` corresponds to Results section 3.2 and figure references therein
- ``examples/example_in_vivo_tetrode.py`` corresponds to Results section 3.3 and figure references therein
- ``examples/example_in_vitro_MEA/example_in_vitro_MEA.py`` corresponds to Results-section 3.4 and figure references therein
