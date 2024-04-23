#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/printUtilities.h"
#include "include/initUtilities.h"
#include "include/MMMutilities.h" 

#define USECBLAS
#define DEBUG

void buildRecvCountsAndDispls(int* recvcounts, int* displs, int NPEs, int N, int colID)
{
    int nCols = N/NPEs + (colID < N % NPEs ? 1 : 0);
    for(int j=0; j<NPEs; j++){
        int nRows= N/NPEs + (j < N % NPEs ? 1 : 0);
        recvcounts[j] = nRows*nCols;
        displs[j] = j>0 ? displs[j-1] + recvcounts[j-1] : 0;
    }
}

int main(int argc, char** argv)
{
    const uint N = atoi(argv[1]);
    int myPID, NPEs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myPID);
    MPI_Comm_size(MPI_COMM_WORLD, &NPEs);
    const uint workSize = N/NPEs;
    const uint workSizeRemainder = N % NPEs;
    const uint myWorkSize = workSize + (myPID < workSizeRemainder ? 1 : 0);
    double* myA = (double*)malloc(myWorkSize*N*sizeof(double));
    double* myB = (double*)malloc(myWorkSize*N*sizeof(double));
    double* myC = (double*)malloc(myWorkSize*N*sizeof(double));
    memset(myC, 0, myWorkSize*N*sizeof(double));

    initID(myA, myPID, N, myWorkSize, NPEs);
    init(myB, myPID, N, myWorkSize, NPEs);

    #ifdef DEBUG
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
        nColumnsBblock = workSize + (i < workSizeRemainder ? 1 : 0);
        startPoint = i*workSize + (i < workSizeRemainder ? i : workSizeRemainder);
        myBblock = (double*)malloc(myWorkSize*nColumnsBblock*sizeof(double));
        readBlockFromMatrix(myBblock, myB, myWorkSize, nColumnsBblock, N, startPoint);
        columnB = (double*)malloc(nColumnsBblock*N*sizeof(double));
        buildRecvCountsAndDispls(recvcounts, displs, NPEs, N, i);
        MPI_Allgatherv(myBblock, myWorkSize*nColumnsBblock, MPI_DOUBLE, columnB, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        
        #ifdef DEBUG
            printMatrixDistributed(columnB, myWorkSize, N, myPID, NPEs);
            printf("\n\n");
        #endif
        #ifdef USECBLAS
            matMulCblas(myA, columnB, myC, myWorkSize, N, nColumnsBblock, startPoint);
        #else
            matMul(myA, columnB, myC, myWorkSize, N, nColumnsBblock, startPoint);
        #endif
        free(myBblock);
        free(columnB);
    }
    MPI_Barrier(MPI_COMM_WORLD); 

    #ifdef DEBUG
        printMatrixDistributed(myC, myWorkSize, N, myPID, NPEs);
    #endif

    free(myA);
    free(myB);
    free(myC);
    free(recvcounts);
    free(displs);
    MPI_Finalize();
    return 0;
}