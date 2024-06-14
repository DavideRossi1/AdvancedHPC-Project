/**
 * @file utilities.h
 * @author Davide Rossi
 * @brief Header file for some utility functions
 * @date 2024-06
 * 
 */
#pragma once
#include <stdlib.h>
#include "timer.h"

// Guard to avoid compiler to compile something as C++ code
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Multiply two matrices A and B with the triple loop algorithm and store the result in the right position in C
 * 
 * @param A first matrix
 * @param B second matrix
 * @param C output matrix
 * @param nRowsA number of rows of the first matrix
 * @param nColsARowsB number of columns of the first matrix and rows of the second matrix
 * @param nColsB number of columns of the second matrix
 * @param startingCol starting point for the current process to store the result in C
 */
void naiveMult(double *A, double *B, double *C, uint nRowsA, uint nColsARowsB, uint nColsB, uint startingCol);


/**
 * @brief Read a block from a particular position of a given matrix
 * 
 * @param block the block to read
 * @param matrix the matrix to read from
 * @param nBlockRows number of rows of the block
 * @param nBlockCols number of columns of the block
 * @param nMatrixCols number of columns of the matrix
 * @param startingCol starting point of the block inside the matrix
 */
void readBlockFromMatrix(double *block, double *matrix, uint nBlockRows, uint nBlockCols, uint nMatrixCols, uint startingCol);


/**
 * @brief Place a block in a particular position of a given matrix
 * 
 * @param block the block to place
 * @param matrix the matrix to place the block in
 * @param nBlockRows number of rows of the block
 * @param nBlockCols number of columns of the block
 * @param nMatrixCols number of columns of the matrix
 * @param startingCol starting point of the block inside the matrix
 */
void placeBlockInMatrix(double *block, double *matrix, uint nBlockRows, uint nBlockCols, uint nMatrixCols, uint startingCol);


/**
 * @brief Build the recvcounts and displs arrays for the MPI_Allgatherv function
 * 
 * @param recvcounts array of the number of elements to receive from each process
 * @param displs array of the displacements where to store the received elements
 * @param NPEs total number of MPI processes
 * @param N total number of columns of the matrix, to be divided among the processes
 * @param colID current iteration number
 */
void buildRecvCountsAndDispls(int* recvcounts, int* displs, uint NPEs, uint N, uint colID);

#ifdef __cplusplus
}
#endif
