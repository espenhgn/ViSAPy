#!/bin/sh
#PBS -lnodes=4:ppn=16
#PBS -lwalltime=0:20:00
#PBS -A nn4661k

cd $PBS_O_WORKDIR
mpirun -np 32 -bynode -bind-to-core -cpus-per-proc 2 python ISI_waveforms.py
wait
