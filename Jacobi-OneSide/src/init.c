/**
 * @file init.c
 * @author Davide Rossi
 * @brief Source file for the init function
 * @date 2024-06
 * 
 */
#include <mpi.h>
#include <omp.h>

#include "init.h"

void init(double* matrix, double* matrix_new, size_t nRows, size_t nCols, int prev, int next, uint shift)
{
  //fill initial values  
#pragma omp parallel for collapse(2)
  for(size_t i = 0; i < nRows; ++i )
    for(size_t j = 1; j < nCols-1; ++j ) {
      matrix[ i*nCols + j ] = 0.5;   
      matrix_new[ i*nCols + j ] = 0.0;
    }
  // set up borders 
  double increment = 100.0/(nCols-1);
  // fill the first and last column
#pragma omp parallel for
  for(size_t i = 0; i < nRows; ++i ){
    matrix[ i*nCols ] = (i+shift)*increment;
    matrix_new[ i*nCols ] = (i+shift)*increment;
    matrix[ (i+1)*nCols - 1 ] = 0.0;
    matrix_new[ (i+1)*nCols - 1 ] = 0.0;
  }
  // if you are the first process, fill the first row with 0
  if (prev == MPI_PROC_NULL){
#pragma omp parallel for
    for(size_t i = 0; i < nCols; ++i)
      matrix[i] = 0.0;
  }
  // if you are the last process, fill the last row
  if (next == MPI_PROC_NULL){
#pragma omp parallel for
    for(size_t i = 1; i < nCols; ++i ){
      matrix[ nRows*nCols-1-i ] = i*increment;
      matrix_new[ nRows*nCols-1-i ] = i*increment;
    }
  }
}
