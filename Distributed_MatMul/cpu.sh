#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="cpu-matMul"
#SBATCH --account ict24_dssc_cpu
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH --gres=tmpfs:10g
#SBATCH --time=02:00:00

module load openmpi/4.1.6--gcc--12.2.0
module load openblas/0.3.24--gcc--12.2.0

echo "Running on $SLURM_NNODES nodes"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

size=5000
file=data/naive$size.csv
make clean
make naive
echo "Init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nNodes in 1 2 4 8 16
        do mpirun -np $nNodes ./main $size >> $file
done

file=data/cpu$size.csv
make clean
make cpu
echo "Init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nNodes in 1 2 4 8 16
do
        do mpirun -np $nNodes ./main $size >> $file
done

echo "Done"
