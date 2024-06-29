#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="gpu-matMul"
#SBATCH --account ict24_dssc_gpu
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=8
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=481G
#SBATCH --time=00:30:00

module load openmpi/4.1.6--nvhpc--23.11

echo "Running on $SLURM_NNODES nodes"

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=8

make clean
make gpu

size=5000
file=data/gpu$size.csv
echo "initCuda;init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nTasks in 1 2 4 8 16 32
do
        mpirun -np $nTasks --map-by ppr:4:node:pe=8 --bind-to core ./main $size >> $file
done

size=10000
file=data/gpu$size.csv
echo "initCuda;init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nTasks in 1 2 4 8 16 32
do
        mpirun -np $nTasks --map-by ppr:4:node:pe=8 --bind-to core ./main $size >> $file
done

size=45000
file=data/gpu$size.csv
echo "initCuda;init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nTasks in 1 2 4 8 16 32
do
        mpirun -np $nTasks --map-by ppr:4:node:pe=8 --bind-to core ./main $size >> $file
done

echo "Done"
