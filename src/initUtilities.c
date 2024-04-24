#include <string.h>

#include "../include/initUtilities.h"

void initOrder(double* matrix, uint myRank, uint N, uint workSize, uint NPEs)
{
    uint shift = 0;
    for (uint i = 0; i < myRank; i++)
        shift += N*(N/NPEs + (i < N % NPEs ? 1 : 0));
    for (uint i = 0; i < workSize*N; i++)
        matrix[i] = i + shift;
}

void initID(double* matrix, uint myRank, uint N, uint workSize, uint NPEs)
{
    memset(matrix, 0, workSize*N*sizeof(double));
    uint shift = myRank*workSize + (myRank < N%NPEs ? 0 : N%NPEs);
    for (uint i = 0; i < workSize; i++) 
        matrix[i*N + shift + i] = 1;
}

void initRandom(double* matrix, uint myRank, uint N, uint workSize, uint NPEs)
{
    for (uint i = 0; i < workSize*N; i++)
        matrix[i] = (double)rand() / RAND_MAX;
}



