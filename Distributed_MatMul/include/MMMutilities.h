#pragma once
#include <stdlib.h>
#include "timings.h"

void readBlockFromMatrix(double *block, double *matrix, size_t nRows, size_t nCols, size_t N, size_t startingCol);

void placeBlockInMatrix(double *block, double *matrix, size_t nRows, size_t nCols, size_t N, size_t startingCol);

void matMul(double *A, double *B, double *C, size_t nRowsA, size_t nColsARowsB, size_t nColsB, size_t startingCol, struct Timings* timings);
