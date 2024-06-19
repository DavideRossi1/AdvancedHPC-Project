/**
 * @file init.c
 * @author Davide Rossi
 * @brief Source file for the init function
 * @date 2024-06
 * 
 */
#include <omp.h>

#include "init.h"

void init(double* matrix, double* matrix_new, size_t nRows, size_t nCols, double* firstRow, double* lastRow, uint shift, uint myRank, uint NPEs)
{
  //fill initial values  
#pragma omp parallel
{
#pragma omp for collapse(2)
  for(size_t i = 0; i < nRows; ++i )
    for(size_t j = 1; j < nCols-1; ++j ) {
      matrix[ i*nCols + j ] = 0.5; 
      matrix_new[ i*nCols + j ] = 0.0;
    }
  // set up borders 
  double increment = 100.0/(nCols-1);
  // fill the first and last column
#pragma omp for
  for(size_t i = 0; i < nRows; ++i ){
    matrix[ i*nCols ] = (i+shift+1)*increment;
    matrix_new[ i*nCols ] = (i+shift+1)*increment;
    matrix[ (i+1)*nCols - 1 ] = 0.0;
    matrix_new[ (i+1)*nCols - 1 ] = 0.0;
  }

  if (myRank) {
    firstRow[0] = shift*increment;
    firstRow[nCols-1] = 0.0;
#pragma omp for
    for (size_t i = 1; i < nCols-1; i++) firstRow[i] = 0.5;
  } else {
#pragma omp for
    for (size_t i = 0; i < nCols; i++) firstRow[i] = 0.0;
  }

  if (myRank < NPEs-1) {
    lastRow[0] = (nRows+shift+1)*increment;
    lastRow[nCols-1] = 0.0;
#pragma omp for
    for (size_t i = 1; i < nCols-1; i++) lastRow[i] = 0.5;
  } else {
#pragma omp for
    for(size_t i = 0; i < nCols; ++i ) lastRow[nCols-1-i] = i*increment;
  }
}
}
