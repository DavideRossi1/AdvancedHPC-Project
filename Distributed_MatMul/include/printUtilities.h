#pragma once
#include <stdlib.h>

void printMatrixThrSafe(double *matrix, size_t nRows, size_t nCols, uint myRank, uint NPEs);

void convertMatrix(double *matrix, char* charMatrix, size_t nRows, size_t nCols);

void printMatrixDistributed(double *matrix, size_t nRows, size_t nCols, uint myRank, uint NPEs);

void printMatrix(double *matrix, size_t nRows, size_t nCols);