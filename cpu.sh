#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="cpu-test"
#SBATCH --account ict24_dssc_cpu
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH --gres=tmpfs:10g
#SBATCH --time=04:00:00

module load openmpi/4.1.6--gcc--12.2.0
module load openblas/0.3.24--gcc--12.2.0

echo "Running on $SLURM_NNODES nodes"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

make clean
make
for nNodes in 1 2 4 8 16
do
	for i in {1..10}
		do mpirun -np $nNodes ./main 5000 >> basic5000.csv
	done
done

make clean
make blas
for nNodes in 1 2 4 8 16
do
        for i in {1..10}
                do mpirun -np $nNodes ./main 5000 >> blas5000.csv
        done
done

echo "Done"
