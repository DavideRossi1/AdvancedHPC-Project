#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="blas-test"
#SBATCH --account ict24_dssc_cpu
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --gres=tmpfs:10g
#SBATCH --time=00:15:00

module load openmpi/4.1.6--gcc--12.2.0
module load openblas/0.3.24--gcc--12.2.0
module load cuda

echo "Running on $SLURM_NNODES nodes"

make clean
make cuda

export OMP_PLACES=cores
export OMP_PROC_BIND=close

mpirun -np 4 ./main 10 >> output.csv

echo "Done"
