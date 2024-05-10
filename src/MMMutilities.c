#include <string.h>
#include <omp.h>
#ifdef CBLAS
  #include <cblas.h>
#endif

#ifdef CUDA
  #include <cublas_v2.h>
  #include <cuda_runtime.h>
#endif

#include "../include/MMMutilities.h"

void readBlockFromMatrix(double *block, double *matrix, uint nBlockRows, uint nBlockCols, uint nMatrixCols, uint startingCol) 
{
#pragma omp parallel for collapse(2)
  for (uint i = 0; i < nBlockRows; i++)
    for (uint j = 0; j < nBlockCols; j++)
      block[i * nBlockCols + j] = matrix[i * nMatrixCols + j + startingCol];
}

void placeBlockInMatrix(double *block, double *matrix, uint nBlockRows, uint nBlockCols, uint nMatrixCols, uint startingCol) 
{
#pragma omp parallel for collapse(2)
  for (uint i = 0; i < nBlockRows; i++)
    for (uint j = 0; j < nBlockCols; j++)
      matrix[i * nMatrixCols + j + startingCol] = block[i * nBlockCols + j];
}

void matMul(double *A, double *B, double *C, uint nRowsA, uint nColsARowsB, uint nColsB, uint startingCol) 
{
  #ifdef CUDA
    cublasHandle_t handle;
    cublasCreate(&handle);
    double *myCBlock = (double *)malloc(nRowsA * nColsB * sizeof(double));
    memset(myCBlock, 0, nRowsA * nColsB * sizeof(double));
    const double alpha = 1.0;
    const double beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nColsB, nRowsA,
        nColsARowsB, &alpha, B, nColsARowsB, A, nColsARowsB, &beta, myCBlock, nColsB);
    placeBlockInMatrix(myCBlock, C, nRowsA, nColsB, nColsARowsB, startingCol);
    free(myCBlock);
    cublasDestroy(handle);  
  #else
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
  #endif
}
