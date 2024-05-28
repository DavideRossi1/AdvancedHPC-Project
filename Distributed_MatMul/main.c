#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>
#ifdef CUDA
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
#endif

#include "printUtilities.h"
#include "initUtilities.h"
#include "MMMutilities.h"

void buildRecvCountsAndDispls(size_t* recvcounts, size_t* displs, uint NPEs, size_t N, size_t colID);

void initAndPrintMatrices(double* myA, double* myB, double* myC, size_t N, size_t myWorkSize, uint myRank, uint NPEs);

int main(int argc, char** argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: ./main <N>\n");
        return 1;
    }
    const size_t N = atoi(argv[1]);
    int myRank, NPEs, provided;
    MPI_Init_thread(&argc, &argv,MPI_THREAD_MULTIPLE,&provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &NPEs);
    struct Timings timings = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; 
    timings.programStart = MPI_Wtime();
    
    #ifdef CUDA
        int NGPUS=-1;
        cudaGetDeviceCount(&NGPUS);
        #ifdef DEBUG
            if(!myRank) printf("Running with %d GPUs\n",NGPUS);
        #endif
        cudaSetDevice(myRank % NGPUS);
    #endif
    const size_t workSize = N/NPEs;
    const size_t workSizeRemainder = N % NPEs;
    const size_t myWorkSize = workSize + ((size_t)myRank < workSizeRemainder ? 1 : 0);

    MPI_Barrier(MPI_COMM_WORLD);
    timings.start = MPI_Wtime();
    double* myA = (double*)malloc(N*myWorkSize*sizeof(double));
    double* myB = (double*)malloc(N*myWorkSize*sizeof(double));
    double* myC = (double*)malloc(N*myWorkSize*sizeof(double));
    initAndPrintMatrices(myA, myB, myC, N, myWorkSize, myRank, NPEs);
    timings.initTime = MPI_Wtime() - timings.start;
    
    size_t* recvcounts = (size_t*)malloc(NPEs*sizeof(size_t));
    size_t* displs = (size_t*)malloc(NPEs*sizeof(size_t));
    double* myBblock;
    double* columnB;
    uint nColumnsBblock, startPoint;
    #ifdef CUDA
        double* A_dev;
        cudaMalloc((void**) &A_dev, myWorkSize*N*sizeof(double));
        cudaMemcpy(A_dev, myA, myWorkSizeuint*N*sizeof(double), cudaMemcpyHostToDevice);
    #endif

    for(uint i = 0; i < (uint)NPEs; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        timings.start = MPI_Wtime();
        nColumnsBblock = workSize + (i < workSizeRemainder ? 1 : 0);
        startPoint = i*workSize + (i < workSizeRemainder ? i : workSizeRemainder);
        myBblock = (double*)malloc(nColumnsBblock*myWorkSize*sizeof(double));
        readBlockFromMatrix(myBblock, myB, myWorkSize, nColumnsBblock, N, startPoint);
        columnB = (double*)malloc(nColumnsBblock*N*sizeof(double));
        buildRecvCountsAndDispls(recvcounts, displs, NPEs, N, i);
        timings.initCommTime += MPI_Wtime() - timings.start;
        timings.start = MPI_Wtime();
        MPI_Allgatherv(myBblock, myWorkSize*nColumnsBblock, MPI_DOUBLE, columnB, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        timings.gatherTime += MPI_Wtime() - timings.start;
        timings.multStart = MPI_Wtime();
        #ifdef CUDA
            matMul(A_dev, columnB, myC, myWorkSize, N, nColumnsBblock, startPoint,&timings);
        #else
            matMul(myA, columnB, myC, myWorkSize, N, nColumnsBblock, startPoint,&timings);
        #endif
        timings.multTime += MPI_Wtime() - timings.multStart;
        free(myBblock);
        free(columnB);
    } 
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef DEBUG
        printMatrixThrSafe(myC, myWorkSize, N, myRank, NPEs);
    #endif
    #ifdef CUDA
        cudaFree(A_dev);
    #endif
    free(myA);
    free(myB);
    free(myC);
    free(recvcounts);
    free(displs);
    timings.totalTime = MPI_Wtime() - timings.programStart;
    #ifdef PRINTTIME
        printTimings(&timings, myRank, NPEs);
    #endif
    MPI_Finalize();
    return 0;
}

void buildRecvCountsAndDispls(size_t* recvcounts, size_t* displs, uint NPEs, size_t N, size_t colID)
{
    size_t nCols = N/NPEs + (colID < N % NPEs ? 1 : 0);
    for(uint j=0; j<NPEs; j++){
        size_t nRows= N/NPEs + (j < N % NPEs ? 1 : 0);
        recvcounts[j] = nRows*nCols;
        displs[j] = j > 0 ? displs[j-1] + recvcounts[j-1] : 0;
    }
}

void initAndPrintMatrices(double* myA, double* myB, double* myC, size_t N, size_t myWorkSize, uint myRank, uint NPEs)
{
    memset(myC, 0, myWorkSize*N*sizeof(double));
    #ifdef DEBUG
        initID(myA, myRank, N, myWorkSize, NPEs);
        initOrder(myB, myRank, N, myWorkSize, NPEs);
        printMatrixThrSafe(myA, myWorkSize, N, myRank, NPEs);
        printMatrixThrSafe(myB, myWorkSize, N, myRank, NPEs);
    #else
        initRandom(myA, N*myWorkSize);
        initRandom(myB, N*myWorkSize);
    #endif
}
