#!/bin/sh
#PBS -lnodes=4:ppn=16
#PBS -lwalltime=6:00:00

cd $PBS_O_WORKDIR
mpirun -np 64 python example_in_vitro_MEA.py
wait