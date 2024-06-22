#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="cpu-Jacobi"
#SBATCH --account ict24_dssc_cpu
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --time=00:40:00

module load openmpi/4.1.6--nvhpc--23.11

echo "Running on $SLURM_NNODES nodes"

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=112

make clean
make

size=1200
nIter=10
file=data/cpu$size.csv

echo "initPar;init;update;comm;evolve;save;total" >> $file
for nTasks in 1 2
do
        mpirun -np $nTasks --map-by node --bind-to none ./main $size $nIter >> $file
done

size=12000
nIter=10
file=data/cpu$size.csv

echo "initPar;init;update;comm;evolve;save;total" >> $file
for nTasks in 1 2
do
       mpirun -np $nTasks --map-by node --bind-to none ./main $size $nIter >> $file
done


size=40000
nIter=1000
file=data/cpu$size.csv

echo "initPar;init;update;comm;evolve;save;total" >> $file
for nTasks in 1 2
do
        mpirun -np $nTasks --map-by node --bind-to none ./main $size $nIter >> $file
done

echo "Done"
