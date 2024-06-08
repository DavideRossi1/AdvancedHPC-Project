#pragma once
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Print the matrix in a thread-safe way: build a string that is printed by the master process
void printMatrixThrSafe(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs);

// Convert a matrix of doubles to a matrix of characters
void convertMatrix(double *matrix, char* charMatrix, uint nRows, uint nCols);

// Print the matrix in a distributed way: each process sends its part to the master process, which prints the entire matrix
void printMatrixDistributed(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs);

// Print a matrix in a sequential way
void printMatrix(double *matrix, uint nRows, uint nCols);

#ifdef __cplusplus
}
#endif
