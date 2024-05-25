#include <mpi.h>
#include <omp.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENACC
  #include <accel.h>
#endif

#include "printUtilities.h"
#include "initUtilities.h"
#include "evolveUtilities.h"
#include "timings.h"

int main(int argc, char* argv[])
{  
  // check on input parameters
  if(argc != 3) {
    fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it\n");
    return 1;
  }
  int myRank, NPEs,provided;
  MPI_Init_thread(&argc, &argv,MPI_THREAD_MULTIPLE,&provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &NPEs);  

  struct Timings timings = { 0, 0, 0, 0, 0, 0 };
  MPI_Barrier(MPI_COMM_WORLD);
  timings.programStart = MPI_Wtime();

  #ifdef _OPENACC
    const int ngpu = acc_get_num_devices(acc_device_nvidia);
    const int gpuid = myRank % ngpu;
    acc_set_device_num(gpuid, acc_device_nvidia);
    acc_init(acc_device_nvidia);
    if( !myRank ) fprintf(stdout, "NUM GPU: %d\n", ngpu);
    fprintf(stdout, "GPU ID: %d, PID: %d\n", gpuid, myRank);
    fflush( stdout );
  #endif

  size_t dim = atoi(argv[1]);
  size_t iterations = atoi(argv[2]);
  size_t dimWithBord = dim + 2;

  const uint workSize = dim/NPEs;
  const uint workSizeRemainder = dim % NPEs;
  const uint myWorkSize = workSize + ((uint)myRank < workSizeRemainder ? 1 : 0) + 2;  // 2 rows added for the borders
  size_t my_byte_dim = sizeof(double) * myWorkSize * dimWithBord;
  double *matrix     = ( double* )malloc( my_byte_dim ); 
  double *matrix_new = ( double* )malloc( my_byte_dim );
  double *tmp_matrix;
  #pragma acc enter data create(matrix[:myWorkSize*dimWithBord], matrix_new[:myWorkSize*dimWithBord])
  MPI_Barrier(MPI_COMM_WORLD);
  timings.start = MPI_Wtime();
#pragma acc data copy(myRank, NPEs, ngpu, gpuid, dim, iterations, dimWithBord, workSize, workSizeRemainder, myWorkSize, timings)
{
  init( matrix, matrix_new, myWorkSize, dimWithBord, myRank, NPEs);
  MPI_Barrier(MPI_COMM_WORLD);
  #ifdef PRINTMATRIX
    #pragma acc update self(matrix[:myWorkSize*dimWithBord])
    printMatrixDistributed(matrix, myWorkSize-2, dimWithBord, myRank, NPEs);
    MPI_Barrier(MPI_COMM_WORLD);
    if(!myRank) printf("\n");
    #pragma acc update self(matrix_new[:myWorkSize*dimWithBord])
    printMatrixDistributed(matrix_new, myWorkSize-2, dimWithBord, myRank, NPEs);
    MPI_Barrier(MPI_COMM_WORLD);
  #endif
  timings.initTime = MPI_Wtime() - timings.start;
  // start algorithm
  timings.start = MPI_Wtime();
  for(size_t it = 0; it < iterations; ++it ){
    evolve( matrix, matrix_new, myWorkSize, dimWithBord, myRank, NPEs);
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef PRINTMATRIX
      printMatrixDistributed(matrix, myWorkSize-2, dimWithBord, myRank, NPEs);
      if(!myRank) printf("\n");
      MPI_Barrier(MPI_COMM_WORLD);
    #endif
    tmp_matrix = matrix;
    matrix = matrix_new;
    matrix_new = tmp_matrix;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  timings.evolveTime = MPI_Wtime() - timings.start;

  // save results
  timings.start = MPI_Wtime();
  save_gnuplot( matrix, dimWithBord, myRank, NPEs, myWorkSize);
  MPI_Barrier(MPI_COMM_WORLD);
  timings.saveTime = MPI_Wtime() - timings.start;

  #ifdef CONVERT
    convertBinToTxt();
  #endif
  free( matrix );
  free( matrix_new );
  timings.totalTime = MPI_Wtime() - timings.programStart;

  #ifdef PRINTTIME
    printTimings(&timings, myRank, NPEs);
  #endif
}
  MPI_Finalize();
  return 0;
}
