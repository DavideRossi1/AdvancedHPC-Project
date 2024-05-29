#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

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
    if ( !ngpu ) {
      fprintf(stderr, "No NVIDIA GPU found\n");
      return 1;
    }
    const int gpuid = myRank % ngpu;
    acc_set_device_num(gpuid, acc_device_nvidia);
    acc_init(acc_device_nvidia);
  #endif

  size_t dim = atoi(argv[1]);
  size_t iterations = atoi(argv[2]);
  size_t dimWithEdge = dim + 2;
  int prev = myRank ? myRank-1 : MPI_PROC_NULL;
  int next = myRank != NPEs-1 ? myRank+1 : MPI_PROC_NULL;
  const uint workSize = dim/NPEs;
  const uint workSizeRemainder = dim % NPEs;
  const uint myWorkSize = workSize + ((uint)myRank < workSizeRemainder ? 1 : 0) + 2;  // 2 rows added for the borders
  size_t my_byte_dim = sizeof(double) * myWorkSize * dimWithEdge;
  double *matrix     = ( double* )malloc( my_byte_dim ); 
  double *matrix_new = ( double* )malloc( my_byte_dim );
  double *tmp_matrix;
  start(&timings);
#pragma acc data copy(timings, matrix[:myWorkSize*dimWithEdge], matrix_new[:myWorkSize*dimWithEdge])
{
  init( matrix, matrix_new, myWorkSize, dimWithEdge, myRank, NPEs, prev, next);
  timings.initTime = end(&timings);
  #ifdef DEBUG
    printMatrixThrSafe(matrix, myWorkSize, dimWithEdge, myRank, NPEs);
    printMatrixThrSafe(matrix_new, myWorkSize, dimWithEdge, myRank, NPEs);
  #endif
  // start algorithm
  start(&timings);
  for(size_t it = 0; it < iterations; ++it )
  {
    evolve( matrix, matrix_new, myWorkSize, dimWithEdge, prev, next);
    MPI_Barrier(MPI_COMM_WORLD);
    tmp_matrix = matrix;
    matrix = matrix_new;
    matrix_new = tmp_matrix;
    #ifdef DEBUG
      printMatrixThrSafe(matrix, myWorkSize, dimWithEdge, myRank, NPEs);
    #endif
  }
  timings.evolveTime = end(&timings);

  // save results
  start(&timings);
  save_gnuplot( matrix, dimWithEdge, myRank, NPEs, myWorkSize);
  timings.saveTime = end(&timings);
  
  MPI_Barrier(MPI_COMM_WORLD);
  timings.totalTime = MPI_Wtime() - timings.programStart;

  #ifdef PRINTTIME
    printTimings(&timings, myRank, NPEs);
  #endif
}
  free( matrix );
  free( matrix_new );
  MPI_Finalize();
  return 0;
}
