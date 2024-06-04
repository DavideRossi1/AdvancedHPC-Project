#pragma once
#include <stdlib.h>

// Initialize a matrix with increasing values. Rank and NPEs are used to compute the correct starting value
void initOrder(double* matrix, uint nRows, uint nCols, uint myRank, uint NPEs);

// Initialize a matrix with the identity matrix. Rank and NPEs are used to compute the correct starting value
void initID(double* matrix, uint nRows, uint nCols, uint myRank, uint NPEs);

// Initialize a matrix with random values
void initRandom(double* matrix, size_t nElements);

// Initialize and optionally print the three given matrices
void initAndPrintMatrices(double* myA, double* myB, double* myC, uint nRows, uint nCols, uint myRank, uint NPEs);
