#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="cpu-matMul"
#SBATCH --account ict24_dssc_cpu
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH --gres=tmpfs:10g
#SBATCH --time=01:00:00

module load openmpi/4.1.6--nvhpc--23.11
module load openblas/0.3.24--nvhpc--23.11

echo "Running on $SLURM_NNODES nodes"

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=112

make clean
make naive

size=5000
file=data/naive$size.csv
echo "initCuda;init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nNodes in 8 16
        do mpirun -np $nNodes --map-by node --bind-to none ./main $size >> $file
done

size=10000
file=data/naive$size.csv
echo "initCuda;init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nNodes in 8 16
        do mpirun -np $nNodes --map-by node --bind-to none ./main $size >> $file
done


make clean
make cpu

size=5000
file=data/cpu$size.csv
echo "initCuda;init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nNodes in 8 16
        do mpirun -np $nNodes --map-by node --bind-to none ./main $size >> $file
done

size=10000
file=data/cpu$size.csv
echo "initCuda;init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nNodes in 8 16
        do mpirun -np $nNodes --map-by node --bind-to none ./main $size >> $file
done

size=30000
file=data/cpu$size.csv
echo "initCuda;init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nNodes in 8 16
        do mpirun -np $nNodes --map-by node --bind-to none ./main $size >> $file
done

size=45000
file=data/cpu$size.csv
echo "initCuda;init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nNodes in 8 16
        do mpirun -np $nNodes --map-by node --bind-to none ./main $size >> $file
done


echo "Done"
