#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="gpu-test"
#SBATCH --account ict24_dssc_gpu
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=64
#SBATCH --ntasks=256                  
#SBATCH --ntasks-per-node=4          
#SBATCH --cpus-per-task=8            
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=00:15:00

module load openmpi/4.1.6--nvhpc--23.11

echo "Running on $SLURM_NNODES nodes"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

make clean
make cuda
echo "Init;initComm;gather;resAlloc;dGemm;place;mult;total" >> gpu80000.csv
for nNodes in 1 2 4 8 16 32 64
do
        for i in {1..5}
                do mpirun -np $nNodes ./main 80000 >> gpu80000.csv
        done
done

echo "Done"
