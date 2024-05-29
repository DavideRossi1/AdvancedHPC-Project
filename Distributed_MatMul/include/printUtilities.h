#pragma once
#include <stdlib.h>

void printMatrixThrSafe(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs);

void convertMatrix(double *matrix, char* charMatrix, uint nRows, uint nCols);

void printMatrixDistributed(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs);

void printMatrix(double *matrix, uint nRows, uint nCols);