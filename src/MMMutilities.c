#include <stdlib.h>
#include <string.h>
#include <cblas.h>

#include "../include/MMMutilities.h"

void readBlockFromMatrix(double *block, double *matrix, int nRows,
                                int nCols, int N, int startingCol) {
  for (int i = 0; i < nRows; i++)
    for (int j = 0; j < nCols; j++)
      block[i * nCols + j] = matrix[i * N + j + startingCol];
}
void placeBlockInMatrix(double *block, double *matrix, int nRows,
                               int nCols, int N, int startingCol) {
  for (int i = 0; i < nRows; i++)
    for (int j = 0; j < nCols; j++)
      matrix[i * N + j + startingCol] = block[i * nCols + j];
}
void matMul(double *A, double *B, double *C, int nRowsA, int nColsARowsB,
            int nColsB, int startingCol) {
  for (int i = 0; i < nRowsA; i++) {
    for (int j = 0; j < nColsB; j++) {
      for (int k = 0; k < nColsARowsB; k++)
        C[i * nColsARowsB + startingCol + j] +=
            A[i * nColsARowsB + k] * B[k * nColsB + j];
    }
  }
}
void matMulCblas(double *A, double *B, double *C, int nRowsA, int nColsARowsB,
                 int nColsB, int startingCol) {
  double *myCBlock = (double *)malloc(nRowsA * nColsB * sizeof(double));
  memset(myCBlock, 0, nRowsA * nColsB * sizeof(double));
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nRowsA, nColsB,
              nColsARowsB, 1.0, A, nColsARowsB, B, nColsB, 1.0, myCBlock,
              nColsB);
  placeBlockInMatrix(C, myCBlock, startingCol, nRowsA, nColsB, nColsARowsB);
  free(myCBlock);
}
