#include <mpi.h>

#include "init.h"


void init(double* matrix, double* matrix_new, size_t nRows, size_t nCols, int prev, int next, uint shift)
{
#pragma acc parallel loop present(matrix[:nRows*nCols], matrix_new[:nRows*nCols])
  for(size_t k = 0; k< nRows*nCols; k++){ 
    matrix[k] = 0.0;
    matrix_new[k] = 0.0;
  }
  //fill initial values  
  //fill the inner square
#pragma acc parallel loop collapse(2) present(matrix[:nRows*nCols], matrix_new[:nRows*nCols])
  for(size_t i = 1; i < nRows-1; ++i )
    for(size_t j = 1; j < nCols-1; ++j )
      matrix[ i*nCols + j ] = 0.5;   
  // set up borders 
  double increment = 100.0/(nCols-1);
  // fill the first column
#pragma acc parallel loop present(matrix[:nRows*nCols], matrix_new[:nRows*nCols])
  for(size_t i = 1; i < nRows-1; ++i ){
    matrix[ i*nCols ] = (i+shift)*increment;
    matrix_new[ i*nCols ] = (i+shift)*increment;
  }
  // fill the last row
  if (next == MPI_PROC_NULL){
#pragma acc parallel loop present(matrix[:nRows*nCols], matrix_new[:nRows*nCols])
    for(size_t i = 1; i < nCols; ++i ){
      matrix[ nRows*nCols-1-i ] = i*increment;
      matrix_new[ nRows*nCols-1-i ] = i*increment;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Sendrecv(&matrix[nCols], nCols, MPI_DOUBLE, prev, 1, 
               &matrix[0],     nCols, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&matrix_new[nCols], nCols, MPI_DOUBLE, prev, 2, 
               &matrix_new[0],     nCols, MPI_DOUBLE, prev, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&matrix[(nRows-2)*nCols], nCols, MPI_DOUBLE, next, 0, 
               &matrix[(nRows-1)*nCols], nCols, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&matrix_new[(nRows-2)*nCols], nCols, MPI_DOUBLE, next, 3, 
               &matrix_new[(nRows-1)*nCols], nCols, MPI_DOUBLE, next, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
