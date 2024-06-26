#include <stdio.h>
#include <mpi.h>

#ifdef _OPENACC
  #include <accel.h>
#endif

#include "print.h"
#include "init.h"
#include "evolve.h"
#include "timer.h"

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

  struct Timer t = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  MPI_Barrier(MPI_COMM_WORLD);
  t.programStart = MPI_Wtime();

#ifdef _OPENACC
  start(&t);
  const int ngpu = acc_get_num_devices(acc_device_nvidia);
  if ( !ngpu ) {
    fprintf(stderr, "No NVIDIA GPU found\n");
    return 1;
  }
  const int gpuid = myRank % ngpu;
  acc_set_device_num(gpuid, acc_device_nvidia);
  acc_init(acc_device_nvidia);
  t.initACC = end(&t);
#endif

  size_t dim = atoi(argv[1]);
  size_t iterations = atoi(argv[2]);
  size_t dimWithEdge = dim + 2;
  int prev = myRank ? myRank-1 : MPI_PROC_NULL;
  int next = myRank != NPEs-1 ? myRank+1 : MPI_PROC_NULL;

  const uint workSize = dim/NPEs;
  const uint workSizeRemainder = dim % NPEs;
  const uint myWorkSize = workSize + ((uint)myRank < workSizeRemainder ? 1 : 0) + 2;  // 2 rows added for the borders
  const size_t my_byte_dim = sizeof(double) * myWorkSize * dimWithEdge;
  const uint shift = myRank*workSize + ((uint)myRank < workSizeRemainder ? (uint)myRank : workSizeRemainder);
#if defined(SAVEPLOT) || defined(SAVEGIF)
  // For plot, skip the first and last row, except for the first and last process
  uint firstRow = myRank ? 1 : 0;
  uint lastRow = myRank < NPEs-1 ? myWorkSize-1 : myWorkSize;
#endif
  double *matrix     = ( double* )malloc( my_byte_dim ); 
  double *matrix_new = ( double* )malloc( my_byte_dim );
  double *tmp_matrix;  

  start(&t);
#pragma acc data create(matrix[:myWorkSize*dimWithEdge], matrix_new[:myWorkSize*dimWithEdge]) copyout(matrix[:myWorkSize*dimWithEdge])
{
  t.copyin = end(&t);
  start(&t);
  init( matrix, matrix_new, myWorkSize, dimWithEdge, prev, next, shift);
  t.init = end(&t);
#ifdef DEBUG
#pragma acc update self(matrix[:myWorkSize*dimWithEdge], matrix_new[:myWorkSize*dimWithEdge])
  printMatrixThrSafe(matrix_new, myWorkSize, dimWithEdge, myRank, NPEs);
  printMatrixThrSafe(matrix, myWorkSize, dimWithEdge, myRank, NPEs);
#endif
#ifdef SAVEGIF
#pragma acc update self(matrix[:myWorkSize*dimWithEdge]) 
  save_gnuplot( matrix, dimWithEdge, firstRow, lastRow, shift, 0);
#endif
  // start algorithm
  MPI_Barrier(MPI_COMM_WORLD);
  t.evolveStart = MPI_Wtime();
  for(size_t it = 0; it < iterations; ++it )
  {
    evolve( matrix, matrix_new, myWorkSize, dimWithEdge, prev, next, &t);
    MPI_Barrier(MPI_COMM_WORLD);
    tmp_matrix = matrix;
    matrix = matrix_new;
    matrix_new = tmp_matrix;
#ifdef DEBUG
#pragma acc update self(matrix[:myWorkSize*dimWithEdge]) 
    printMatrixThrSafe(matrix, myWorkSize, dimWithEdge, myRank, NPEs);
#endif
#ifdef SAVEGIF
#pragma acc update self(matrix[:myWorkSize*dimWithEdge]) 
    save_gnuplot( matrix, dimWithEdge, firstRow, lastRow, shift, it+1);
#endif
  }
  MPI_Barrier(MPI_COMM_WORLD);
  t.evolve = MPI_Wtime() - t.evolveStart;
  start(&t);
}
  t.copyout = end(&t);
#ifdef SAVEPLOT
  // save results
  start(&t);
  save_gnuplot( matrix, dimWithEdge, firstRow, lastRow, shift, 0);
  t.save = end(&t);
#endif
  t.total = MPI_Wtime() - t.programStart;
  printTimings(&t, myRank, NPEs);
  free( matrix );
  free( matrix_new );
  MPI_Finalize();
  return 0;
}
