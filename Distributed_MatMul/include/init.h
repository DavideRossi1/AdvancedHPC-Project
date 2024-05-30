#pragma once
#include <stdlib.h>

void initOrder(double* matrix, uint nRows, uint nCols, uint myRank, uint NPEs);

void initID(double* matrix, uint nRows, uint nCols, uint myRank, uint NPEs);

void initRandom(double* matrix, size_t nElements);

void initAndPrintMatrices(double* myA, double* myB, double* myC, uint nRows, uint nCols, uint myRank, uint NPEs);
