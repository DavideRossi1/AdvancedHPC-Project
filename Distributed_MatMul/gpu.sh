#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="gpu-matMul"
#SBATCH --account ict24_dssc_gpu
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=2
#SBATCH --ntasks=8
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

size=5000
file=data/gpu$size.csv

make clean
make gpu

echo "initCuda;init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nTasks in 1 2
do
        mpirun -np $nTasks --map-by ppr:4:node --bind-to core ./main $size >> $file
done

size=10000
file=data/gpu$size.csv
echo "initCuda;init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nTasks in 1 2
do
        mpirun -np $nTasks --map-by ppr:4:node --bind-to core ./main $size >> $file
done

size=45000
file=data/gpu$size.csv
echo "initCuda;init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nTasks in 1 2
do
        mpirun -np $nTasks --map-by ppr:4:node --bind-to core ./main $size >> $file
done

echo "Done"
