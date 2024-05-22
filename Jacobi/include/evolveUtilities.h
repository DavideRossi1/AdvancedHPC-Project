#pragma once
#include <stdlib.h>

void evolve( double * matrix, double *matrix_new, size_t nRows, size_t nCols, int myRank, int NPEs);
