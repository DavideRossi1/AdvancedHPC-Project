#include <string.h>
#include <omp.h>

#include "initUtilities.h"

void initOrder(double* matrix, uint myRank, uint N, uint workSize, uint NPEs)
{
    size_t shift = myRank*N*(N/NPEs) + N*(myRank < N % NPEs ? myRank : N % NPEs);
#pragma omp parallel for
    for (size_t i = 0; i < workSize*N; i++)
        matrix[i] = i + shift;
}

void initID(double* matrix, uint myRank, uint N, uint workSize, uint NPEs)
{
    memset(matrix, 0, workSize*N*sizeof(double));
    size_t shift = myRank*workSize + (myRank < N%NPEs ? 0 : N%NPEs);
#pragma omp parallel for
    for (uint i = 0; i < workSize; i++) 
        matrix[i*N + shift + i] = 1;
}

void initRandom(double* matrix, size_t nElements)
{
#pragma omp parallel for
    for (size_t i = 0; i < nElements; i++)
        matrix[i] = (double)rand() / RAND_MAX;
}

void initAndPrintMatrices(double* myA, double* myB, double* myC, uint N, uint myWorkSize, int myRank, int NPEs)
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
