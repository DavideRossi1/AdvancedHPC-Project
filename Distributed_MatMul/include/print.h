/**
 * @file print.h
 * @author Davide Rossi
 * @brief Header file for the print functions
 * @date 2024-06
 * 
 */
#pragma once
#include <stdlib.h>

// Guard to avoid compiler to compile something as C++ code
#ifdef __cplusplus
extern "C" {
#endif

/**
* @brief Print the matrix in a distributed and thread-safe way: build a string that is eventually printed by the master process

* @param matrix the matrix to print
* @param nRows number of rows of the matrix
* @param nCols number of columns of the matrix
* @param myRank MPI rank of the executing process
* @param NPEs total number of MPI processes
*/
void printMatrixThrSafe(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs);


/**
 * @brief Convert a matrix of doubles to a matrix of characters
 * 
 * @param matrix the matrix to convert
 * @param charMatrix the output matrix of characters
 * @param nRows number of rows of the matrix
 * @param nCols number of columns of the matrix
 */
void convertMatrix(double *matrix, char* charMatrix, uint nRows, uint nCols);


/**
 * @brief Print the matrix in a distributed way: each process sends its part to the master process which prints it
 * 
 * @param matrix the matrix to print
 * @param nRows number of rows of the matrix
 * @param nCols number of columns of the matrix
 * @param myRank MPI rank of the executing process
 * @param NPEs total number of MPI processes
 */
void printMatrixDistributed(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs);


/**
 * @brief Print a matrix in a sequential way
 * 
 * @param matrix the matrix to print
 * @param nRows number of rows of the matrix
 * @param nCols number of columns of the matrix
 */
void printMatrix(double *matrix, uint nRows, uint nCols);

#ifdef __cplusplus
}
#endif
