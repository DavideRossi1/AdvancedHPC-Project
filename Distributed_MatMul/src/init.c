/**
 * @file init.c
 * @author Davide Rossi
 * @brief Source file for the initialization functions
 * @date 2024-06
 * 
 */
#include <omp.h>
#include <string.h>

#include "init.h"

void initOrder(double* matrix, uint nRows, uint nCols, uint myRank, uint NPEs)
{
    size_t shift = nCols * (myRank*(nCols/NPEs) + (myRank < nCols%NPEs ? myRank : nCols%NPEs));
#pragma omp parallel for
    for (size_t i = 0; i < nRows*nCols; i++)
        matrix[i] = i + shift;
}

void initID(double* matrix, uint nRows, uint nCols, uint myRank, uint NPEs)
{
    memset(matrix, 0, nRows*nCols*sizeof(double));
    size_t shift = myRank*nRows + (myRank < nCols%NPEs ? 0 : nCols%NPEs);
#pragma omp parallel for
    for (uint i = 0; i < nRows; i++) 
        matrix[i*nCols + shift + i] = 2;
}

void initRandom(double* matrix, size_t nElements)
{
#pragma omp parallel
{
    uint seed = omp_get_thread_num();
    #pragma omp for
    for (size_t i = 0; i < nElements; i++)
        matrix[i] = (double)rand_r(&seed) / RAND_MAX;
}
}

void initAll(double* myA, double* myB, double* myC, uint nRows, uint nCols, uint myRank, uint NPEs)
{
    memset(myC, 0, nRows*nCols*sizeof(double));
    #ifdef DEBUG
        initID(myA, nRows, nCols,  myRank, NPEs);
        initOrder(myB, nRows, nCols, myRank, NPEs);
    #else
        initRandom(myA, nRows*nCols);
        initRandom(myB, nRows*nCols);
    #endif
}
