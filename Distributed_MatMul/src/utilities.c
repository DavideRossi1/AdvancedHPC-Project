#include <omp.h>
#include <string.h>

#include "utilities.h"

void naiveMult(double *A, double *B, double *C, uint nRowsA, uint nColsARowsB, uint nColsB, uint startingCol) 
{
    for (uint i = 0; i < nRowsA; i++)
      for (uint j = 0; j < nColsB; j++)
        for (uint k = 0; k < nColsARowsB; k++)
          C[i*nColsARowsB + j + startingCol] += A[i*nColsARowsB + k] * B[k*nColsB + j];
}

void readBlockFromMatrix(double *block, double *matrix, uint nBlockRows, uint nBlockCols, uint nMatrixCols, uint startingCol) 
{
#pragma omp parallel for collapse(2)
  for (uint i = 0; i < nBlockRows; i++)
    for (uint j = 0; j < nBlockCols; j++)
      block[i*nBlockCols + j] = matrix[i*nMatrixCols + j + startingCol];
}

void placeBlockInMatrix(double *block, double *matrix, uint nBlockRows, uint nBlockCols, uint nMatrixCols, uint startingCol) 
{
#pragma omp parallel for collapse(2)
  for (uint i = 0; i < nBlockRows; i++)
    for (uint j = 0; j < nBlockCols; j++)
      matrix[i*nMatrixCols + j + startingCol] = block[i*nBlockCols + j];
}

void buildRecvCountsAndDispls(int* recvcounts, int* displs, uint NPEs, uint N, uint colID)
{
    uint nCols = N/NPEs + (colID < N % NPEs ? 1 : 0);
    for(uint j=0; j<NPEs; j++){
      uint nRows= N/NPEs + (j < N % NPEs ? 1 : 0);
      recvcounts[j] = nRows*nCols;
      displs[j] = j > 0 ? displs[j-1] + recvcounts[j-1] : 0;
    }
}