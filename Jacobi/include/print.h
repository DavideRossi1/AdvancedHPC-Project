#pragma once
#include <stdlib.h>

#pragma once
#include <stdlib.h>


// Print the matrix in a thread-safe way: build a string that is printed by the master process
void printMatrixThrSafe(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs);

// Convert a matrix of doubles to a matrix of characters
void convertMatrix(double *matrix, char* charMatrix, uint nRows, uint nCols);

// Get the size of a row in characters (only for debugging)
size_t getRowSize(uint matrixSize);

// Print the matrix in a distributed way: each process sends its part to the master process, which prints the entire matrix
void printMatrixDistributed(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs);

// Print a matrix in a sequential way
void printMatrix(double *matrix, uint nRows, uint nCols);

void save_gnuplot( double *M, size_t dim, uint start, uint end, uint shift, size_t it);
