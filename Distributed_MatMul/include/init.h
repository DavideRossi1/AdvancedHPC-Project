#pragma once
#include <stdlib.h>

// Guard to avoid compiler to compile something as C++ code
#ifdef __cplusplus
extern "C" {
#endif

/**
* Initialize a matrix with increasing values. Rank and NPEs are used to compute the correct starting value
*
* @param matrix the matrix to initialize
* @param nRows number of rows of the matrix
* @param nCols number of columns of the matrix
* @param myRank MPI rank of the executing process
* @param NPEs total number of MPI processes
*/
void initOrder(double* matrix, uint nRows, uint nCols, uint myRank, uint NPEs);

/**
* Initialize a matrix with the identity matrix. Rank and NPEs are used to compute the correct starting value
*
* @param matrix the matrix to initialize
* @param nRows number of rows of the matrix
* @param nCols number of columns of the matrix
* @param myRank MPI rank of the executing process
* @param NPEs total number of MPI processes
*/
void initID(double* matrix, uint nRows, uint nCols, uint myRank, uint NPEs);

/**
* Initialize a matrix with random values
*
* @param matrix the matrix to initialize
* @param nElements the total number of elements (nRows*nCols)
*/
void initRandom(double* matrix, size_t nElements);

/**
* Initialize the three given matrices
*
* @param myA first matrix
* @param myB second matrix
* @param myC output matrix
* @param nRows number of rows of the matrices
* @param nCols number of columns of the matrices
* @param myRank MPI rank of the executing process
* @param NPEs total number of MPI processes
*/
void initAll(double* myA, double* myB, double* myC, uint nRows, uint nCols, uint myRank, uint NPEs);

#ifdef __cplusplus
}
#endif