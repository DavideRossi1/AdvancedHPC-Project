#include <mpi.h>
#include <mpi_proto.h>
#include <omp.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "include/printUtilities.h"
#include "include/initUtilities.h"
#include "include/timings.h"

// evolve Jacobi
void evolve( double * matrix, double *matrix_new, size_t nRows, size_t nCols, int myRank, int NPEs, MPI_Status* status);

int main(int argc, char* argv[])
{  
  // check on input parameters
  if(argc != 3) {
    fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it\n");
    return 1;
  }
  size_t dim = atoi(argv[1]);
  size_t iterations = atoi(argv[2]);
  size_t dimWithBord = dim + 2;

  int myRank, NPEs,provided;
  MPI_Init_thread(&argc, &argv,MPI_THREAD_MULTIPLE,&provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &NPEs);  
  struct Timings timings = { 0, 0, 0, 0, 0, 0};
  MPI_Barrier(MPI_COMM_WORLD);
  timings.programStart = MPI_Wtime();

  const uint workSize = dim/NPEs;
  const uint workSizeRemainder = dim % NPEs;
  const uint myWorkSize = workSize + ((uint)myRank < workSizeRemainder ? 1 : 0) + 2;  // 2 rows added for the borders
  size_t my_byte_dim = sizeof(double) * myWorkSize * dimWithBord;
  double *matrix     = ( double* )malloc( my_byte_dim ); 
  double *matrix_new = ( double* )malloc( my_byte_dim );
  double *tmp_matrix;
  
  // initialize matrix
  MPI_Barrier(MPI_COMM_WORLD);
  timings.start = MPI_Wtime();
  memset( matrix, 0, my_byte_dim );
  memset( matrix_new, 0, my_byte_dim );
  MPI_Status status;
  init( matrix, matrix_new, myWorkSize, dimWithBord, myRank, NPEs, &status);
  MPI_Barrier(MPI_COMM_WORLD);
  #ifdef PRINTMATRIX
    printMatrixDistributed(matrix, myWorkSize, dimWithBord, myRank, NPEs);
    MPI_Barrier(MPI_COMM_WORLD);
    if(!myRank) printf("\n");
    printMatrixDistributed(matrix_new, myWorkSize, dimWithBord, myRank, NPEs);
    MPI_Barrier(MPI_COMM_WORLD);
  #endif
  timings.initTime = MPI_Wtime() - timings.start;
  // start algorithm
  timings.start = MPI_Wtime();
  for(size_t it = 0; it < iterations; ++it ){
    evolve( matrix, matrix_new, myWorkSize, dimWithBord, myRank, NPEs, &status);
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef PRINTMATRIX
      printMatrixDistributed(matrix, myWorkSize, dimWithBord, myRank, NPEs);
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
  save_gnuplot( matrix, dimWithBord, myRank, NPEs, myWorkSize, status);
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
  MPI_Finalize();
  return 0;
}


void evolve( double* matrix, double* matrix_new, size_t nRows, size_t nCols, int myRank, int NPEs, MPI_Status* status)
{
  //This will be a row dominant program.
#pragma omp parallel for collapse(2)
  for(size_t i = 1; i < nRows-1; ++i )
    for(size_t j = 1; j < nCols-1; ++j ) {
      size_t currentEl = i*nCols + j;
      matrix_new[currentEl] = 0.25*( matrix[currentEl-nCols] + matrix[currentEl+1] + 
                                     matrix[currentEl+nCols] + matrix[currentEl-1] );
    }
  MPI_Barrier(MPI_COMM_WORLD);
  #pragma omp master
  {
    if (myRank)
      MPI_Sendrecv(&matrix[nCols], nCols, MPI_DOUBLE, myRank-1, 1, 
                   &matrix[0], nCols, MPI_DOUBLE, myRank-1, 0, MPI_COMM_WORLD, status);
    if (myRank != NPEs-1)
      MPI_Sendrecv(&matrix[(nRows-2)*nCols], nCols, MPI_DOUBLE, myRank+1, 0, 
                   &matrix[(nRows-1)*nCols], nCols, MPI_DOUBLE, myRank+1, 1, MPI_COMM_WORLD, status);
  }
}