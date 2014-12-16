#!/bin/sh
#PBS -lnodes=1:ppn=16
#PBS -lwalltime=24:00:00

cd $PBS_O_WORKDIR
mpirun -np 6 -bynode -bind-to-core -cpus-per-proc 2 python example_in_vivo_tetrode.py --quiet
wait