#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef CUDA
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
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
    int myRank, NPEs,provided;
    MPI_Init_thread(&argc, &argv,MPI_THREAD_MULTIPLE,&provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &NPEs);
    #ifdef PRINTTIME
	if(!myRank) printf("Init;Comm;Mult\n");
    #endif
    #ifdef CUDA
        int NGPUS;
        cudaGetDeviceCount(&NGPUS);
        if(!myRank) printf("Running with %d GPUs\n",NGPUS);
        cudaSetDevice(myRank % NGPUS);
    #endif
    double maxInitTime, maxCommTime, maxMultTime;
    double programStart, start;
    double commTime = 0;
    double multTime = 0;
    const uint workSize = N/NPEs;
    const uint workSizeRemainder = N % NPEs;
    const uint myWorkSize = workSize + ((uint)myRank < workSizeRemainder ? 1 : 0);

    MPI_Barrier(MPI_COMM_WORLD);
    programStart = MPI_Wtime();
    double* myA = (double*)malloc(N*myWorkSize*sizeof(double));
    double* myB = (double*)malloc(N*myWorkSize*sizeof(double));
    double* myC = (double*)malloc(N*myWorkSize*sizeof(double));
    initAndPrintMatrices(myA, myB, myC, N, myWorkSize, myRank, NPEs);
    double initTime = MPI_Wtime() - programStart;
    MPI_Reduce(&initTime, &maxInitTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    #ifdef PRINTTIME
        if(!myRank) printf("%f;", maxInitTime);
    #endif
    
    int* recvcounts = (int*)malloc(NPEs*sizeof(int));
    int* displs = (int*)malloc(NPEs*sizeof(int));
    double* myBblock;
    double* columnB;
    uint nColumnsBblock, startPoint;
    #ifdef CUDA
        double *myA_dev;
        cudaMalloc((void **) &myA_dev, myWorkSize*N*sizeof(double));
        cudaMemcpy(myA_dev, myA, myWorkSize*N*sizeof(double), cudaMemcpyHostToDevice);
    #endif
    for(uint i = 0; i < (uint)NPEs; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();
        nColumnsBblock = workSize + (i < workSizeRemainder ? 1 : 0);
        startPoint = i*workSize + (i < workSizeRemainder ? i : workSizeRemainder);
        myBblock = (double*)malloc(nColumnsBblock*myWorkSize*sizeof(double));
        readBlockFromMatrix(myBblock, myB, myWorkSize, nColumnsBblock, N, startPoint);
        columnB = (double*)malloc(nColumnsBblock*N*sizeof(double));
        buildRecvCountsAndDispls(recvcounts, displs, NPEs, N, i);
        MPI_Allgatherv(myBblock, myWorkSize*nColumnsBblock, MPI_DOUBLE, columnB, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        commTime += MPI_Wtime() - start;
        start = MPI_Wtime();
        matMul(myA, columnB, myC, myWorkSize, N, nColumnsBblock, startPoint);
        multTime += MPI_Wtime() - start;
        free(myBblock);
        free(columnB);
    } 
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef PRINTMATRIX
        printMatrixDistributed(myC, myWorkSize, N, myRank, NPEs);
    #endif
    MPI_Reduce(&commTime, &maxCommTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&multTime, &maxMultTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    #ifdef PRINTTIME
        if(!myRank){
            printf("%f;", maxCommTime);
            printf("%f\n", maxMultTime);
        }
    #endif
    free(myA);
    free(myB);
    free(myC);
    free(recvcounts);
    free(displs);
    MPI_Finalize();
    return 0;
}
