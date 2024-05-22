#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="gpu-test"
#SBATCH --account ict24_dssc_gpu
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=32
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=02:00:00

module load openmpi/4.1.6--nvhpc--23.11

echo "Running on $SLURM_NNODES nodes"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

make clean
make cuda
echo "init;initComm;gather;resAlloc;dGemm;place;mult;total" >> data/gpu80000.csv
for nTasks in 1 2 4 8 16 32 64 128
do
        echo nTasks >> data/gpu80000.csv
        for i in {1..3}
                do mpirun -np $nTasks ./main 80000 >> data/gpu80000.csv
        done
done

echo "Done"
