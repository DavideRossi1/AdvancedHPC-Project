#pragma once
#include <stdlib.h>
#include "timings.h"

void matMul(double *A, double *B, double *C, uint nRowsA, uint nColsARowsB, uint nColsB, uint startingCol, struct Timings* timings);

void readBlockFromMatrix(double *block, double *matrix, uint nRows, uint nCols, uint N, uint startingCol);

void placeBlockInMatrix(double *block, double *matrix, uint nRows, uint nCols, uint N, uint startingCol);

