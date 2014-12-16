
In order to be able to run this particular example, follow the steps below:

Compile cython extension corresponding to the MoI (Method of Images) class in
this folder
::
    
    python setup.py --build_ext -i
    
Compile NEURON NMODL language files in folder modfiles
::
    
    cd modfiles
    nrnivmodl
    cd -

Make sure that all dependencies are met in the main simulation script
example_in_vitro_MEA.py, and run it on a cluster, e.g., calling
::
    
    mpirun -np 64 python example_in_vitro_MEA.py

However, it is more likely better to adapt the corresponding jobscript to your
needs, and submit to the que
::
    
    qsub example_in_vitro_MEA.sh
