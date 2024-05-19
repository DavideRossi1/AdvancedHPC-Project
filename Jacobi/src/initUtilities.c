#include <stdio.h>
#include "../include/initUtilities.h"

void init(double* matrix, double* matrix_new, size_t nRows, size_t nCols, uint myRank, uint NPEs, MPI_Status* status)
{
  //fill initial values  
  //fill the inner square
#pragma omp parallel for collapse(2)
  for(size_t i = 1; i < nRows-1; ++i )
    for(size_t j = 1; j < nCols-1; ++j )
      matrix[ i*nCols + j ] = 0.5;   
   
  // set up borders 
  uint shift = myRank*((nCols-2)/NPEs) + (myRank < (nCols-2)%NPEs ? myRank : (nCols-2)%NPEs);
  #ifdef DEBUG
    printf("proc %d, shift %d\n", myRank, shift);
  #endif
  double increment = 100.0/(nCols-1);
  // fill the first column
#pragma omp parallel for
  for(size_t i = 1; i < nRows-1; ++i ){
    matrix[ i*nCols ] = (i+shift)*increment;
    matrix_new[ i*nCols ] = (i+shift)*increment;
  }
  // fill the last row
  if (myRank == NPEs-1){
#pragma omp parallel for
    for(size_t i = 1; i < nCols; ++i ){
      matrix[ nRows*nCols-1-i ] = i*increment;
      matrix_new[ nRows*nCols-1-i ] = i*increment;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (NPEs == 1) return;
  #pragma omp master
  {
    if (myRank){
        MPI_Sendrecv(&matrix[nCols], nCols, MPI_DOUBLE, myRank-1, 1, 
                     &matrix[0],     nCols, MPI_DOUBLE, myRank-1, 0, MPI_COMM_WORLD, status);
        MPI_Sendrecv(&matrix_new[nCols], nCols, MPI_DOUBLE, myRank-1, 2, 
                     &matrix_new[0], nCols, MPI_DOUBLE, myRank-1, 3, MPI_COMM_WORLD, status);
    }
    if (myRank != NPEs-1){
        MPI_Sendrecv(&matrix[(nRows-2)*nCols], nCols, MPI_DOUBLE, myRank+1, 0, 
                     &matrix[(nRows-1)*nCols], nCols, MPI_DOUBLE, myRank+1, 1, MPI_COMM_WORLD, status);
        MPI_Sendrecv(&matrix_new[(nRows-2)*nCols], nCols, MPI_DOUBLE, myRank+1, 3, 
                     &matrix_new[(nRows-1)*nCols], nCols, MPI_DOUBLE, myRank+1, 2, MPI_COMM_WORLD, status);
    }
  }
}
