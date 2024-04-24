#include <stdlib.h>

void readBlockFromMatrix(double *block, double *matrix, uint nRows, uint nCols, uint N, uint startingCol);

void placeBlockInMatrix(double *block, double *matrix, uint nRows, uint nCols, uint N, uint startingCol);

void matMul(double *A, double *B, double *C, uint nRowsA, uint nColsARowsB, uint nColsB, uint startingCol);
