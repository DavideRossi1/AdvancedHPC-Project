#pragma once
#include <stdlib.h>

void initOrder(double* matrix, uint myRank, uint N, uint workSize, uint NPEs);

void initID(double* matrix, uint myRank, uint N, uint workSize, uint NPEs);

void initRandom(double* matrix, size_t nElements);

void initAndPrintMatrices(double* myA, double* myB, double* myC, uint N, uint myWorkSize, int myRank, int NPEs);
