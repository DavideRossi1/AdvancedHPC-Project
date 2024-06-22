#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="cpu-matMul"
#SBATCH --account ict24_dssc_cpu
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH --gres=tmpfs:10g
#SBATCH --time=01:00:00

module load openmpi/4.1.6--gcc--12.2.0
module load openblas/0.3.24--gcc--12.2.0

echo "Running on $SLURM_NNODES nodes"

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=112

size=5000
file=data/naive$size.csv
make clean
make naive
echo "initCuda;init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nNodes in 1 2 4 8 16
        do mpirun -np $nNodes --map-by node --bind-to none ./main $size >> $file
done

file=data/cpu$size.csv
make clean
make cpu
echo "initCuda;init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nNodes in 1 2 4 8 16
        do mpirun -np $nNodes --map-by node --bind-to none ./main $size >> $file
done



size=10000
file=data/naive$size.csv
make clean
make naive
echo "initCuda;init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nNodes in 1 2 4 8 16
        do mpirun -np $nNodes --map-by node --bind-to none ./main $size >> $file
done

file=data/cpu$size.csv
make clean
make cpu
echo "initCuda;init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nNodes in 1 2 4 8 16
        do mpirun -np $nNodes --map-by node --bind-to none ./main $size >> $file
done

size=45000
file=data/cpu$size.csv
make clean
make cpu
echo "initCuda;init;initComm;gather;resAlloc;dGemm;place;mult;total" >> $file
for nNodes in 1 2 4 8 16
        do mpirun -np $nNodes --map-by node --bind-to none ./main $size >> $file
done


echo "Done"
