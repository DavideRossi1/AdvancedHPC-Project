#pragma once
#include <stdlib.h>

void initOrder(double* matrix, uint myRank, size_t N, size_t workSize, uint NPEs);

void initID(double* matrix, uint myRank, size_t N, size_t workSize, uint NPEs);

void initRandom(double* matrix, size_t nElements);
