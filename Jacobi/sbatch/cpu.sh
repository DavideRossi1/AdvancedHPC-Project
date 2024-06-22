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

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=112

make clean
make cpu

size=1200
nIter=10
file=data/cpu$size.csv

echo "initacc;copyin;init;update;sendrecv;evolve;save;copyout;total" >> $file
for nTasks in 4 8 16
do
        mpirun -np $nTasks --map-by node --bind-to none ./jacobi.x $size $nIter >> $file
done

size=12000
nIter=10
file=data/cpu$size.csv

echo "initacc;copyin;init;update;sendrecv;evolve;save;copyout;total" >> $file
for nTasks in 4 8 16
do
        mpirun -np $nTasks --map-by node --bind-to none ./jacobi.x $size $nIter >> $file
done

make clean
make cpu

size=40000
nIter=1000
file=data/cpu$size.csv

echo "initacc;copyin;init;update;sendrecv;evolve;save;copyout;total" >> $file
for nTasks in 4 8 16
do
        mpirun -np $nTasks --map-by node --bind-to none ./jacobi.x $size $nIter >> $file
done

echo "Done"
