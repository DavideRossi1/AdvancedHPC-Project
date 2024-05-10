#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="gpu-test"
#SBATCH --account ict24_dssc_gpu
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

module load openmpi/4.1.6--gcc--12.2.0
module load cuda

echo "Running on $SLURM_NNODES nodes"

make clean

export OMP_PLACES=cores
export OMP_PROC_BIND=close

make cuda
for nNodes in 1 2 4 8 16
do
        for i in {1..10}
                do mpirun -np $nNodes ./main 5000 >> blas5000.csv
        done
done

echo "Done"
