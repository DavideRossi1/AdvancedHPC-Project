#include <omp.h>
#include <string.h>

#ifdef CBLAS
  #include <cblas.h>
#endif

#ifdef CUDA
  #include <cuda_runtime.h>
  #include <cublas_v2.h>
#endif

#include "MMmult.h"

void matMul(double *A, double *B, double *C, uint nRowsA, uint nColsARowsB, uint nColsB, uint startingCol, struct Timer* t) 
{
  #ifdef CUDA
    start(t);
    double* B_dev;
    cudaMalloc((void**) &B_dev, nColsARowsB*nColsB*sizeof(double));
    cudaMemcpy(B_dev, B, nColsARowsB*nColsB*sizeof(double), cudaMemcpyHostToDevice);
    
    double* myCBlock_dev;
    cudaMalloc((void**)&myCBlock_dev, nRowsA*nColsB*sizeof(double));
    double *myCBlock = (double*)malloc(nRowsA*nColsB*sizeof(double));
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alpha = 1.0;
    const double beta = 0.0;
    timings->resAllocTime += end(t);
    start(t);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nColsB, nRowsA,
        nColsARowsB, &alpha, B_dev, nColsB, A, nColsARowsB, &beta, myCBlock_dev, nColsB);
    timings->dgemmTime += end(t);
    start(t);
    cudaMemcpy(myCBlock, myCBlock_dev, nRowsA*nColsB*sizeof(double), cudaMemcpyDeviceToHost);
    placeBlockInMatrix(myCBlock, C, nRowsA, nColsB, nColsARowsB, startingCol);
    free(myCBlock);
    cudaFree(B_dev);
    cudaFree(myCBlock_dev);
    cublasDestroy(handle);
    timings->placeTime += end(t);
  #else
  #ifdef CBLAS
    start(t);
    double *myCBlock = (double *)malloc(nRowsA * nColsB * sizeof(double));
    memset(myCBlock, 0, nRowsA * nColsB * sizeof(double));
    t->resAllocTime += end(t);
    start(t);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nRowsA, nColsB,
                nColsARowsB, 1.0, A, nColsARowsB, B, nColsB, 0.0, myCBlock,nColsB);
    t->dgemmTime += end(t);
    start(t);
    placeBlockInMatrix(myCBlock, C, nRowsA, nColsB, nColsARowsB, startingCol);
    free(myCBlock);
    t->placeTime += end(t);
  #else
    for (uint i = 0; i < nRowsA; i++)
      for (uint j = 0; j < nColsB; j++)
        for (uint k = 0; k < nColsARowsB; k++)
          C[i * nColsARowsB + startingCol + j] += A[i * nColsARowsB + k] * B[k * nColsB + j];
  #endif
  #endif
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
