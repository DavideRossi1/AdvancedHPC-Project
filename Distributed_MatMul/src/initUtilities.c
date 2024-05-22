#include <string.h>
#include <omp.h>

#include "initUtilities.h"

void initOrder(double* matrix, uint myRank, uint N, uint workSize, uint NPEs)
{
    uint shift = myRank*N*(N/NPEs) + N*(myRank < N % NPEs ? myRank : N % NPEs);
#pragma omp parallel for
    for (ulong i = 0; i < workSize*N; i++)
        matrix[i] = i + shift;
}

void initID(double* matrix, uint myRank, uint N, uint workSize, uint NPEs)
{
    memset(matrix, 0, workSize*N*sizeof(double));
    uint shift = myRank*workSize + (myRank < N%NPEs ? 0 : N%NPEs);
#pragma omp parallel for
    for (uint i = 0; i < workSize; i++) 
        matrix[i*N + shift + i] = 1;
}

void initRandom(double* matrix, ulong nElements)
{
#pragma omp parallel for
    for (ulong i = 0; i < nElements; i++)
        matrix[i] = (double)rand() / RAND_MAX;
}
