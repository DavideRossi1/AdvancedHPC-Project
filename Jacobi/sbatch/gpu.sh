#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="gpu-Jacobi"
#SBATCH --account ict24_dssc_gpu
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=8
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:30:00

module load openmpi/4.1.6--nvhpc--23.11

echo "Running on $SLURM_NNODES nodes"

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=8

size=1200
nIter=10
file=data/gpu$size.csv

make clean
make gpusave

echo "initacc;copyin;init;update;sendrecv;evolve;save;copyout;total" >> $file
for nTasks in 1 2 4 8 16 32
do
        mpirun -np $nTasks ./jacobi.x $size $nIter >> $file
done

size=12000
nIter=10
file=data/gpu$size.csv

echo "initacc;copyin;init;update;sendrecv;evolve;save;copyout;total" >> $file
for nTasks in 1 2 4 8 16 32
do
        mpirun -np $nTasks ./jacobi.x $size $nIter >> $file
done

make clean
make gpu

size=40000
nIter=1000
file=data/gpu$size.csv

echo "initacc;copyin;init;update;sendrecv;evolve;save;copyout;total" >> $file
for nTasks in 1 2 4 8 16 32
do
        mpirun -np $nTasks ./jacobi.x $size $nIter >> $file
done

echo "Done"
