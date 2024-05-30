#include <stdio.h>
#ifdef CUDA
    #include <cuda_runtime.h>
#endif

#include "print.h"
#include "init.h"
#include "MMmult.h"
#include "timer.h"

void buildRecvCountsAndDispls(int* recvcounts, int* displs, uint NPEs, uint N, uint colID);

//void printArray(int* array, uint size);

//int MPI_Allgatherv_L(const void *sendbuf, size_t sendcount, MPI_Datatype sendtype, void *recvbuf, const size_t *recvcounts, const size_t *displs, MPI_Datatype recvtype, MPI_Comm comm, int myRank, int NPEs);

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
    struct Timer t = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; 
    MPI_Barrier(MPI_COMM_WORLD);
    t.programStart = MPI_Wtime();
    #ifdef CUDA
        int ngpus=-1;
        cudaGetDeviceCount(&ngpus);
        if (!ngpus){
            fprintf(stderr, "No NVIDIA GPU found\n");
            return 1;
        }
        cudaSetDevice(myRank % ngpus);
    #endif
    const uint workSize = N / NPEs;
    const uint workSizeRem = N % NPEs;
    const uint myNRows = workSize + ((uint)myRank < workSizeRem ? 1 : 0);
    const size_t myByteDim = myNRows * N * sizeof(double);

    start(&t);
    double* myA = (double*)malloc(myByteDim);
    double* myB = (double*)malloc(myByteDim);
    double* myC = (double*)malloc(myByteDim);
    initAndPrintMatrices(myA, myB, myC, myNRows, N, myRank, NPEs);
    t.initTime = end(&t);
    #ifdef DEBUG
        printMatrixThrSafe(myA, myNRows, N, myRank, NPEs);
        printMatrixThrSafe(myB, myNRows, N, myRank, NPEs);
    #endif
    int* recvcounts = (int*)malloc(NPEs*sizeof(int));
    int* displs = (int*)malloc(NPEs*sizeof(int));
    double* myBblock;
    double* columnB;
    uint nColumnsBblock, startPoint;
    #ifdef CUDA
        double* A_dev;
        cudaMalloc((void**) &A_dev, myByteDim);
        cudaMemcpy(A_dev, myA, myByteDim, cudaMemcpyHostToDevice);
    #endif

    for(uint i = 0; i < (uint)NPEs; i++)
    {
        start(&t);
        nColumnsBblock = workSize + (i < workSizeRem ? 1 : 0);
        startPoint = i*workSize + (i < workSizeRem ? i : workSizeRem);
        myBblock = (double*)malloc(nColumnsBblock*myNRows*sizeof(double));
        readBlockFromMatrix(myBblock, myB, myNRows, nColumnsBblock, N, startPoint);
        columnB = (double*)malloc(nColumnsBblock*N*sizeof(double));
        buildRecvCountsAndDispls(recvcounts, displs, NPEs, N, i);
        t.initCommTime += end(&t);
        start(&t);
        MPI_Allgatherv(myBblock, myNRows*nColumnsBblock, MPI_DOUBLE, columnB, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        t.gatherTime += end(&t);
        t.multStart = MPI_Wtime();
        #ifdef CUDA
            matMul(A_dev, columnB, myC, myNRows, N, nColumnsBblock, startPoint, &t);
        #else
            matMul(myA, columnB, myC, myNRows, N, nColumnsBblock, startPoint, &t);
        #endif
        MPI_Barrier(MPI_COMM_WORLD);
        t.multTime += MPI_Wtime() - t.multStart;
        free(myBblock);
        free(columnB);
    } 
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef DEBUG
        printMatrixThrSafe(myC, myNRows, N, myRank, NPEs);
    #endif
    #ifdef CUDA
        cudaFree(A_dev);
    #endif
    free(myA);
    free(myB);
    free(myC);
    free(recvcounts);
    free(displs);
    MPI_Barrier(MPI_COMM_WORLD);
    t.totalTime = MPI_Wtime() - t.programStart;
    printTimings(&t, myRank, NPEs);
    MPI_Finalize();
    return 0;
}

void buildRecvCountsAndDispls(int* recvcounts, int* displs, uint NPEs, uint N, uint colID)
{
    uint nCols = N/NPEs + (colID < N % NPEs ? 1 : 0);
    for(uint j=0; j<NPEs; j++){
        uint nRows= N/NPEs + (j < N % NPEs ? 1 : 0);
        recvcounts[j] = nRows*nCols;
        displs[j] = j > 0 ? displs[j-1] + recvcounts[j-1] : 0;
    }
}

// void printArray(int* array, uint size){
//     for(uint i=0; i<size; i++)
//         printf("%d ", array[i]);
//     printf("\n");
// }

// failed test to enlarge allgatherv to allow larger matrices
// int MPI_Allgatherv_L(const void *sendbuf, size_t sendcount, MPI_Datatype sendtype, void *recvbuf, const size_t *recvcounts, const size_t *displs, MPI_Datatype recvtype, MPI_Comm comm, int myRank, int NPEs) {
//     int maxCount = 2;
//     int nChunks = sendcount / maxCount + 1;
//     int remainder = sendcount % maxCount;
//     int chunkSendcount = maxCount;
//     int output;
//     size_t recvcountsLeft[NPEs];
//     size_t displsCopy[NPEs];
//     memcpy(recvcountsLeft, recvcounts, NPEs*sizeof(size_t));
//     memcpy(displsCopy, displs, NPEs*sizeof(size_t));
//     printf("recvcountsLeft:\n");
//     printArray(recvcountsLeft, NPEs);
//     printf("displsCopy:\n");
//     printArray(displsCopy, NPEs);
//     int* chunkRecvcounts = (int*)malloc(NPEs*sizeof(int));
//     for (int i = 0; i < nChunks; i++) {
//         if (i == nChunks - 1 && remainder > 0) chunkSendcount = remainder;
//         for (int j = 0; j < NPEs; j++) {
//             chunkRecvcounts[j] = recvcountsLeft[j] > maxCount ? maxCount : recvcountsLeft[j];
//             recvcountsLeft[j] -= chunkRecvcounts[j];
//             displsCopy[j] += chunkRecvcounts[j];            
//         }
//         printf("chunkRecvcounts:\n");
//         printArray(chunkRecvcounts, NPEs);
//         printf("displsCopy:\n");
//         printArray(displsCopy, NPEs);
//         output = MPI_Allgatherv(sendbuf, chunkSendcount, sendtype, recvbuf, chunkRecvcounts, displsCopy, recvtype, comm);
//         if (output != MPI_SUCCESS) return output;
//     }
//     return MPI_SUCCESS;
// }