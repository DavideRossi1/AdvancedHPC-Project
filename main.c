#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef CUDA
    #include <cuda_runtime.h>
    #include <cuda.h>
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
        displs[j] = j>0 ? displs[j-1] + recvcounts[j-1] : 0;
    }
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
    clock_t programStart, programEnd;
    programStart = clock();
    const uint workSize = N/NPEs;
    const uint workSizeRemainder = N % NPEs;
    const uint myWorkSize = workSize + (myRank < workSizeRemainder ? 1 : 0);
    clock_t start, end;

    start = clock();
    double* myA = (double*)malloc(myWorkSize*N*sizeof(double));
    double* myB = (double*)malloc(myWorkSize*N*sizeof(double));
    double* myC = (double*)malloc(myWorkSize*N*sizeof(double));
    memset(myC, 0, myWorkSize*N*sizeof(double));

    initRandom(myA, myRank, N, myWorkSize, NPEs);
    initRandom(myB, myRank, N, myWorkSize, NPEs);
    MPI_Barrier(MPI_COMM_WORLD);
    end = clock();
    printf("Init time for proc %d: %f\n", myRank, (double)(end-start)/CLOCKS_PER_SEC);
    #ifdef PRINTMATRIX
        printMatrixDistributed(myA, myWorkSize, N, myPID, NPEs);
        printf("\n\n");
        printMatrixDistributed(myB, myWorkSize, N, myPID, NPEs);
        printf("\n\n");
    #endif

    int* recvcounts = (int*)malloc(NPEs*sizeof(int));
    int* displs = (int*)malloc(NPEs*sizeof(int));
    double* myBblock;
    double* columnB;
    uint nColumnsBblock, startPoint;


    for(uint i = 0; i < NPEs; i++)
    {
        start = clock();
        nColumnsBblock = workSize + (i < workSizeRemainder ? 1 : 0);
        startPoint = i*workSize + (i < workSizeRemainder ? i : workSizeRemainder);
        myBblock = (double*)malloc(myWorkSize*nColumnsBblock*sizeof(double));
        readBlockFromMatrix(myBblock, myB, myWorkSize, nColumnsBblock, N, startPoint);
        columnB = (double*)malloc(nColumnsBblock*N*sizeof(double));
        buildRecvCountsAndDispls(recvcounts, displs, NPEs, N, i);
        MPI_Allgatherv(myBblock, myWorkSize*nColumnsBblock, MPI_DOUBLE, columnB, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        end = clock();
        printf("Comm time for proc %d at iter %d: %f\n", myRank, i, (double)(end-start)/CLOCKS_PER_SEC);
        start = clock();
        matMul(myA, columnB, myC, myWorkSize, N, nColumnsBblock, startPoint);
        end = clock();
        printf("Multip time for proc %d at iter %d: %f\n", myRank, i, (double)(end-start)/CLOCKS_PER_SEC);
        free(myBblock);
        free(columnB);
    } 
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef PRINTMATRIX
        printMatrixDistributed(myC, myWorkSize, N, myPID, NPEs);
    #endif

    free(myA);
    free(myB);
    free(myC);
    free(recvcounts);
    free(displs);
    programEnd = clock();
    printf("Total time for proc %d: %f\n", myRank, (double)(programEnd-programStart)/CLOCKS_PER_SEC);
    MPI_Finalize();
    return 0;
}