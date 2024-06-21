/**
 * @file evolve.c
 * @author Davide Rossi
 * @brief Source file for the evolve function
 * @date 2024-06
 * 
 */
#include <mpi.h>
#include <omp.h>

#include "evolve.h"

void evolve( double* matrix, double* matrix_new, size_t nRows, size_t nCols, double* firstRow, double* lastRow)
{
#pragma omp parallel
{
    // Update matrix_new using matrix values
    size_t currentEl;
    // Update the first and last row
  #pragma omp for
    for(size_t j = 1; j < nCols-1; ++j){
      matrix_new[j] = 0.25*( firstRow[j] + matrix[j+1] + matrix[j+nCols] + matrix[j-1] );
      currentEl = (nRows-1)*nCols + j;
      matrix_new[currentEl] = 0.25*( matrix[currentEl-nCols] + matrix[currentEl+1] + 
                                                  lastRow[j] + matrix[currentEl-1] );
    }
    // Update the inner rows
  #pragma omp for collapse(2)
    for(size_t i = 1; i < nRows-1; ++i )
      for(size_t j = 1; j < nCols-1; ++j ) {
        currentEl = i*nCols + j;
        matrix_new[currentEl] = 0.25*( matrix[currentEl-nCols] + matrix[currentEl+1] + 
                                      matrix[currentEl+nCols] + matrix[currentEl-1] );
      }
}
}
