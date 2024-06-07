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

    // allocate columnB, Bblock and Cblock with the maximum space they will require
    double *columnB  = (double*)malloc((workSize+1)*N*sizeof(double));
    double *myBblock = (double*)malloc(maxBlockByteDim);
    double *myCBlock = (double*)malloc(maxBlockByteDim);
    uint nColumnsBblock, startPoint;
    #ifdef CUDA
        double *A_dev, *B_dev, *myCBlock_dev;
        cudaMalloc((void**) &A_dev, myByteDim);
        cudaMemcpy(A_dev, myA, myByteDim, cudaMemcpyHostToDevice);
        cudaMalloc((void**) &B_dev, (workSize+1)*N*sizeof(double));
        cudaMalloc((void**)&myCBlock_dev, maxBlockByteDim);
        cublasHandle_t handle;
        cublasCreate(&handle);
        const double alpha = 1.0;
        const double beta = 0.0;
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
            cudaMemcpy(B_dev, columnB, nColumnsBblock*N*sizeof(double), cudaMemcpyHostToDevice);
            t.resAlloc += end(&t);
            start(&t);
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nColumnsBblock, myNRows,
                    N, &alpha, B_dev, nColumnsBblock, A_dev, N, &beta, myCBlock_dev, nColumnsBblock);
            t.dgemm += end(&t);
            start(&t);
            cudaMemcpy(myCBlock, myCBlock_dev, myNRows*nColumnsBblock*sizeof(double), cudaMemcpyDeviceToHost);
            t.resAlloc += end(&t);
            start(&t);
            placeBlockInMatrix(myCBlock, myC, myNRows, nColumnsBblock, N, startPoint);
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
    #ifdef DEBUG
        printMatrixThrSafe(myC, myNRows, N, myRank, NPEs);
    #endif
    t.total = MPI_Wtime() - t.programStart;
    #ifdef CUDA
        cudaFree(A_dev);
        cudaFree(B_dev);
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