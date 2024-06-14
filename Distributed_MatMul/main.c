/**
 * @file main.c
 * @author Davide Rossi
 * @brief Main function for the distributed matrix multiplication
 * @date 2024-06
 * 
 */
#include <stdio.h>
#ifdef CBLAS
  #include <cblas.h>
#endif
#ifdef CUDA
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
#endif

#include "print.h"
#include "init.h"
#include "utilities.h"
#include "timer.h"

#ifdef CUDA
__global__ void placeBlockInMatrixKernel(double *block, double *matrix, uint nBlockRows, uint nBlockCols, uint nMatrixCols, uint startingCol) 
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nBlockRows && j < nBlockCols)
        matrix[i * nMatrixCols + j + startingCol] = block[i * nBlockCols + j];
}
#endif

int main(int argc, char** argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: ./main <N>\n");
        return 1;
    }
    const uint N = atoi(argv[1]);
    int myRank, NPEs,provided;
    MPI_Init_thread(&argc, &argv,MPI_THREAD_MULTIPLE,&provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &NPEs);
    struct Timer t = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; 
    MPI_Barrier(MPI_COMM_WORLD);
    t.programStart = MPI_Wtime();
    #ifdef CUDA
        start(&t);
        int ngpus=0;
        cudaGetDeviceCount(&ngpus);
        if (!ngpus){
            fprintf(stderr, "No NVIDIA GPU found\n");
            return 1;
        }
        cudaSetDevice(myRank % ngpus);
        t.initCuda = end(&t);
    #endif
    start(&t);
    const uint workSize = N / NPEs;
    const uint workSizeRem = N % NPEs;
    const uint myNRows = workSize + ((uint)myRank < workSizeRem ? 1 : 0);
    const size_t myByteDim       = myNRows * N * sizeof(double);
    const size_t maxBlockByteDim = myNRows * (workSize+1) * sizeof(double);

    double* myA = (double*)malloc(myByteDim);
    double* myB = (double*)malloc(myByteDim);
    double* myC = (double*)malloc(myByteDim);
    t.resAlloc = end(&t);
    start(&t);
    initAll(myA, myB, myC, myNRows, N, myRank, NPEs);
    t.init = end(&t);
    #ifdef DEBUG
        printMatrixThrSafe(myA, myNRows, N, myRank, NPEs);
        printMatrixThrSafe(myB, myNRows, N, myRank, NPEs);
    #endif
    start(&t);
    int *recvcounts = (int*)malloc(NPEs*sizeof(int));
    int *displs     = (int*)malloc(NPEs*sizeof(int));

    // allocate columnB, Bblock and Cblock once, with the maximum space they will require
    double *columnB  = (double*)malloc((workSize+1)*N*sizeof(double));
    double *myBblock = (double*)malloc(maxBlockByteDim);
    double *myCBlock = (double*)malloc(maxBlockByteDim);
    uint nColumnsBblock, startPoint;
    #ifdef CUDA
        double *A_dev, *columnB_dev, *myCBlock_dev, *C_dev;
        cudaMalloc((void**) &A_dev, myByteDim);
        cudaMemcpy(A_dev, myA, myByteDim, cudaMemcpyHostToDevice);
        cudaMalloc((void**) &columnB_dev, (workSize+1)*N*sizeof(double));
        cudaMalloc((void**) &C_dev, myByteDim);
        cudaMalloc((void**)&myCBlock_dev, maxBlockByteDim);
        cublasHandle_t handle;
        cublasCreate(&handle);
        const double alpha = 1.0;
        const double beta = 0.0;
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((myNRows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (workSize + 1 + threadsPerBlock.y - 1) / threadsPerBlock.y);
    #endif
    t.resAlloc += end(&t);

    for(uint i = 0; i < (uint)NPEs; i++)
    {
        start(&t);
        nColumnsBblock = workSize + (i < workSizeRem ? 1 : 0);
        startPoint = i*workSize + (i < workSizeRem ? i : workSizeRem);
        readBlockFromMatrix(myBblock, myB, myNRows, nColumnsBblock, N, startPoint);
        buildRecvCountsAndDispls(recvcounts, displs, NPEs, N, i);
        t.initComm += end(&t);
        start(&t);
        MPI_Allgatherv(myBblock, myNRows*nColumnsBblock, MPI_DOUBLE, columnB, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        t.gather += end(&t);
        t.multStart = MPI_Wtime();
        #ifdef CUDA
            start(&t);
            cudaMemcpy(columnB_dev, columnB, nColumnsBblock*N*sizeof(double), cudaMemcpyHostToDevice);
            t.resAlloc += end(&t);
            start(&t);
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nColumnsBblock, myNRows,
                    N, &alpha, columnB_dev, nColumnsBblock, A_dev, N, &beta, myCBlock_dev, nColumnsBblock);
            t.dgemm += end(&t);
            start(&t);
            placeBlockInMatrixKernel<<<numBlocks, threadsPerBlock>>>(myCBlock_dev, C_dev, myNRows, nColumnsBblock, N, startPoint);
            t.place += end(&t);
        #elif defined(CBLAS)
            start(&t);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, myNRows, nColumnsBblock,
                N, 1.0, myA, N, columnB, nColumnsBblock, 0.0, myCBlock, nColumnsBblock);
            t.dgemm += end(&t);
            start(&t);
            placeBlockInMatrix(myCBlock, myC, myNRows, nColumnsBblock, N, startPoint);
            t.place += end(&t);
        #else
            start(&t);
            naiveMult(myA, columnB, myC, myNRows, N, nColumnsBblock, startPoint);
            t.dgemm += end(&t);
        #endif
        MPI_Barrier(MPI_COMM_WORLD);
        t.mult += MPI_Wtime() - t.multStart;
    } 
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef CUDA
        start(&t);
        cudaMemcpy(myC, C_dev, myByteDim, cudaMemcpyDeviceToHost);
        t.resAlloc += end(&t);
    #endif
    #ifdef DEBUG
        printMatrixThrSafe(myC, myNRows, N, myRank, NPEs);
    #endif
    t.total = MPI_Wtime() - t.programStart;
    #ifdef CUDA
        cudaFree(A_dev);
        cudaFree(columnB_dev);
        cudaFree(C_dev);
        cudaFree(myCBlock_dev);
        cublasDestroy(handle);
    #endif
    free(myA);
    free(myB);
    free(myC);
    free(myBblock);
    free(myCBlock);
    free(columnB);
    free(recvcounts);
    free(displs);
    printTimings(&t, myRank, NPEs);
    MPI_Finalize();
    return 0;
}
