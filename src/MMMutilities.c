#include <string.h>
#include <cblas.h>

#include "../include/MMMutilities.h"

void readBlockFromMatrix(double *block, double *matrix, uint nBlockRows, uint nBlockCols, uint nMatrixCols, uint startingCol) 
{
  for (uint i = 0; i < nBlockRows; i++)
    for (uint j = 0; j < nBlockCols; j++)
      block[i * nBlockCols + j] = matrix[i * nMatrixCols + j + startingCol];
}

void placeBlockInMatrix(double *block, double *matrix, uint nBlockRows, uint nBlockCols, uint nMatrixCols, uint startingCol) 
{
  for (uint i = 0; i < nBlockRows; i++)
    for (uint j = 0; j < nBlockCols; j++)
      matrix[i * nMatrixCols + j + startingCol] = block[i * nBlockCols + j];
}

void matMul(double *A, double *B, double *C, uint nRowsA, uint nColsARowsB, uint nColsB, uint startingCol) 
{
  #ifdef CBLAS
  double *myCBlock = (double *)malloc(nRowsA * nColsB * sizeof(double));
  memset(myCBlock, 0, nRowsA * nColsB * sizeof(double));
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nRowsA, nColsB,
              nColsARowsB, 1.0, A, nColsARowsB, B, nColsB, 1.0, myCBlock,nColsB);
  placeBlockInMatrix(myCBlock, C, nRowsA, nColsB, nColsARowsB, startingCol);
  free(myCBlock);
  #else
  for (uint i = 0; i < nRowsA; i++)
    for (uint j = 0; j < nColsB; j++)
      for (uint k = 0; k < nColsARowsB; k++)
        C[i * nColsARowsB + startingCol + j] += A[i * nColsARowsB + k] * B[k * nColsB + j];
  #endif
}