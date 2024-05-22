#include <mpi.h>
#include "evolveUtilities.h"

inline void evolve( double* matrix, double* matrix_new, size_t nRows, size_t nCols, int myRank, int NPEs)
{
  //This will be a row dominant program.
#pragma acc parallel loop collapse(2) present(matrix[:nRows*nCols], matrix_new[:nRows*nCols])
  for(size_t i = 1; i < nRows-1; ++i )
    for(size_t j = 1; j < nCols-1; ++j ) {
      size_t currentEl = i*nCols + j;
      matrix_new[currentEl] = 0.25*( matrix[currentEl-nCols] + matrix[currentEl+1] + 
                                     matrix[currentEl+nCols] + matrix[currentEl-1] );
    }
  MPI_Barrier(MPI_COMM_WORLD);
  int prev = myRank ? myRank-1 : MPI_PROC_NULL;
  int next = myRank != NPEs-1 ? myRank+1 : MPI_PROC_NULL;
  #pragma omp master
  {
    MPI_Sendrecv(&matrix_new[nCols], nCols, MPI_DOUBLE, prev, 1, 
                 &matrix_new[0],     nCols, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&matrix_new[(nRows-2)*nCols], nCols, MPI_DOUBLE, next, 0, 
                 &matrix_new[(nRows-1)*nCols], nCols, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}