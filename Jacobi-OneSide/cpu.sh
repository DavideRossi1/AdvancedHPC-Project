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

make clean
make save

size=1200
nIter=10
file=data/cpu$size.csv

echo "initPar;init;update;comm;evolve;save;total" >> $file
for nTasks in 8 16
do
        for i in {1..3}
        do
                echo $nTasks >> $file
                srun -N $nTasks ./main $size $nIter >> $file
        done
done

size=12000
nIter=10
file=data/cpu$size.csv

echo "initPar;init;update;comm;evolve;save;total" >> $file
for nTasks in 8 16
do
        for i in {1..3}
        do    
                srun -N $nTasks ./main $size $nIter >> $file
        done
done


echo "Done"
