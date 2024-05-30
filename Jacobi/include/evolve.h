#pragma once
#include <stdlib.h>
#include "timer.h"

void evolve( double* matrix, double* matrix_new, size_t nRows, size_t nCols, int prev, int next, struct Timer* t);
