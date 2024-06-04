#pragma once
#include <stdlib.h>

// Initialize a matrix for the Jacobi method
void init(double* matrix, double* matrix_new, size_t nRows, size_t nCols, int prev, int next, uint shift);