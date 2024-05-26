#pragma once
#include <stdlib.h>

void init(double* matrix, double* matrix_new, size_t nRows, size_t nCols, uint myRank, uint NPEs, int prev, int next);