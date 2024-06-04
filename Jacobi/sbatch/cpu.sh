#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="cpu-Jacobi"
#SBATCH --account ict24_dssc_cpu
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH --time=00:40:00

module load openmpi/4.1.6--nvhpc--23.11

echo "Running on $SLURM_NNODES nodes"

make clean
make cpu

size=1200
nIter=10
file=data/cpu$size.csv

echo "initacc;copyin;init;update;sendrecv;evolve;save;copyout;total" >> $file
for nTasks in 1 2 4 8 16
do
        echo $nTasks >> $file
        mpirun -np $nTasks ./jacobi.x $size $nIter >> $file
done

size=12000
nIter=10
file=data/cpu$size.csv

echo "initacc;copyin;init;update;sendrecv;evolve;save;copyout;total" >> $file
for nTasks in 1 2 4 8 16
do
        echo $nTasks >> $file
        mpirun -np $nTasks ./jacobi.x $size $nIter >> $file
done


echo "Done"
