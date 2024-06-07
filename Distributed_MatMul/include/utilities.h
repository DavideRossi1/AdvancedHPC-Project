#pragma once
#include <stdlib.h>
#include "timer.h"

// Multiply two matrices A and B and store the result in the right position in C
void naiveMult(double *A, double *B, double *C, uint nRowsA, uint nColsARowsB, uint nColsB, uint startingCol);

// Read a block from a particular position of a given matrix
void readBlockFromMatrix(double *block, double *matrix, uint nRows, uint nCols, uint N, uint startingCol);

// Place a block in a particular position of a given matrix
void placeBlockInMatrix(double *block, double *matrix, uint nRows, uint nCols, uint N, uint startingCol);

void buildRecvCountsAndDispls(int* recvcounts, int* displs, uint NPEs, uint N, uint colID);
