#pragma once
#include <stdlib.h>
#include "timer.h"

// Evolve the matrix using the Jacobi method
void evolve( double* matrix, double* matrix_new, size_t nRows, size_t nCols, int prev, int next, struct Timer* t);
