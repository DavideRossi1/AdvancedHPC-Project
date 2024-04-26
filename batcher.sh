#!/bin/bash
#SBATCH --account ict24_dssc_cpu
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=256
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=00:30:00

module load openmpi/4.1.6--gcc--12.2.0
module load openblas/0.3.24--gcc--12.2.0

echo "Running on $SLURM_NNODES nodes"

make clean
make blas

export OMP_PLACES=cores
export OMP_PROC_BIND=close

mpirun -np 4 ./blas 10000 >> output.csv

echo "Done"