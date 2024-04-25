#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef CUDA
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #define NGPUS 8
#endif


#include "include/printUtilities.h"
#include "include/initUtilities.h"
#include "include/MMMutilities.h" 

void buildRecvCountsAndDispls(int* recvcounts, int* displs, uint NPEs, uint N, uint colID)
{
    uint nCols = N/NPEs + (colID < N % NPEs ? 1 : 0);
    for(uint j=0; j<NPEs; j++){
        uint nRows= N/NPEs + (j < N % NPEs ? 1 : 0);
        recvcounts[j] = nRows*nCols;
        displs[j] = j > 0 ? displs[j-1] + recvcounts[j-1] : 0;
    }
}

void initAndPrintMatrices(double* myA, double* myB, double* myC, uint N, uint myWorkSize, int myRank, int NPEs)
{
    memset(myC, 0, myWorkSize*N*sizeof(double));
    #ifdef DEBUG
        initID(myA, myRank, N, myWorkSize, NPEs);
        initOrder(myB, myRank, N, myWorkSize, NPEs);
    #else
        initRandom(myA, N*myWorkSize);
        initRandom(myB, N*myWorkSize);
    #endif
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef PRINTMATRIX
        printMatrixDistributed(myA, myWorkSize, N, myRank, NPEs);
        printf("\n");
        printMatrixDistributed(myB, myWorkSize, N, myRank, NPEs);
        printf("\n");
    #endif
}

int main(int argc, char** argv)
{
    const uint N = atoi(argv[1]);
    int myRank, NPEs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &NPEs);
    #ifdef CUDA
        int myPID = myRank % NGPUS;
        cudaSetDevice(myPID);
    #endif
    clock_t programStart, start, end;
    double elapsedTime;
    programStart = clock();
    const uint workSize = N/NPEs;
    const uint workSizeRemainder = N % NPEs;
    const uint myWorkSize = workSize + ((uint)myRank < workSizeRemainder ? 1 : 0);

    start = clock();
    double* myA = (double*)malloc(N*myWorkSize*sizeof(double));
    double* myB = (double*)malloc(N*myWorkSize*sizeof(double));
    double* myC = (double*)malloc(N*myWorkSize*sizeof(double));
    initAndPrintMatrices(myA, myB, myC, N, myWorkSize, myRank, NPEs);
    end = clock();
    elapsedTime = (double)(end-start)/CLOCKS_PER_SEC;
    #ifdef PRINTTIME
        printf("Init time for proc %d: %f\n", myRank, elapsedTime);
    #endif
    
    int* recvcounts = (int*)malloc(NPEs*sizeof(int));
    int* displs = (int*)malloc(NPEs*sizeof(int));
    double* myBblock;
    double* columnB;
    uint nColumnsBblock, startPoint;
    #ifdef CUDA
        double* myA_dev, *columnB_dev, *myC_dev;
        cudaMalloc((void**)&myA_dev, myWorkSize*N*sizeof(double));
        cudaMalloc((void**)&myC_dev, myWorkSize*N*sizeof(double));
        cudaMemcpy(myA_dev, myA, myWorkSize*N*sizeof(double), cudaMemcpyHostToDevice);
    #endif
    for(uint i = 0; i < (uint)NPEs; i++)
    {
        start = clock();
        nColumnsBblock = workSize + (i < workSizeRemainder ? 1 : 0);
        startPoint = i*workSize + (i < workSizeRemainder ? i : workSizeRemainder);
        myBblock = (double*)malloc(nColumnsBblock*myWorkSize*sizeof(double));
        readBlockFromMatrix(myBblock, myB, myWorkSize, nColumnsBblock, N, startPoint);
        columnB = (double*)malloc(nColumnsBblock*N*sizeof(double));
        buildRecvCountsAndDispls(recvcounts, displs, NPEs, N, i);
        MPI_Allgatherv(myBblock, myWorkSize*nColumnsBblock, MPI_DOUBLE, columnB, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        end = clock();
        elapsedTime = (double)(end-start)/CLOCKS_PER_SEC;
        #ifdef PRINTTIME
            printf("Comm time for proc %d at iter %d: %f\n", myRank, i, elapsedTime);
        #endif
        #ifdef CUDA
            cudaMalloc((void**)&columnB_dev, nColumnsBblock*N*sizeof(double));
            cudaMemcpy(columnB_dev, columnB, nColumnsBblock*N*sizeof(double), cudaMemcpyHostToDevice);
            start = clock();
            matMul(myA_dev, columnB_dev, myC_dev, myWorkSize, N, nColumnsBblock, startPoint);
            end = clock();
            cudaFree(myB_dev);
        #else
            start = clock();
            matMul(myA, columnB, myC, myWorkSize, N, nColumnsBblock, startPoint);
            end = clock();
        #endif
        elapsedTime = (double)(end-start)/CLOCKS_PER_SEC;
        #ifdef PRINTTIME
            printf("Multip time for proc %d at iter %d: %f\n", myRank, i, elapsedTime);
        #endif
        free(myBblock);
        free(columnB);
    } 
    #ifdef CUDA
        cudaMemcpy(myC, myC_dev, myWorkSize*N*sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(myA_dev);
        cudaFree(myC_dev);
    #endif
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef PRINTMATRIX
        printMatrixDistributed(myC, myWorkSize, N, myRank, NPEs);
    #endif

    free(myA);
    free(myB);
    free(myC);
    free(recvcounts);
    free(displs);
    end = clock();
    elapsedTime = (double)(end-programStart)/CLOCKS_PER_SEC;
    #ifdef PRINTTIME
        printf("Total time for proc %d: %f\n", myRank, elapsedTime);
    #endif
    MPI_Finalize();
    return 0;
}