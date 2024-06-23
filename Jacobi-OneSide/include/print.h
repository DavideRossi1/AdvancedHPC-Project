/**
 * @file print.h
 * @author Davide Rossi
 * @brief Header file for the print and save functions
 * @date 2024-06
 * 
 */
#pragma once
#include <stdlib.h>
#include <mpi.h>

/**
* @brief Print the matrix in a distributed and thread-safe way: build a string that is printed by the master process

* @param matrix the matrix to print
* @param nRows number of rows of the matrix
* @param nCols number of columns of the matrix
* @param myRank MPI rank of the executing process
* @param NPEs total number of MPI processes
* @param firstRow the first row of the matrix
* @param lastRowWin the window containing the last row of the matrix
*/
void printMatrixThrSafe(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs, double* firstRow, MPI_Win lastRowWin);


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
 * @param firstRow the first row of the matrix
 * @param lastRowWin the window containing the last row of the matrix
 */
void printMatrixDistributed(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs, double* firstRow, MPI_Win lastRowWin);


/**
 * @brief Print a matrix in a sequential way
 * 
 * @param matrix the matrix to print
 * @param nRows number of rows of the matrix
 * @param nCols number of columns of the matrix
 */
void printMatrix(double *matrix, uint nRows, uint nCols);


/**
 * @brief Save the matrix in a file to be plotted with gnuplot
 * 
 * @param M the matrix to save
 * @param nRows number of rows of the matrix
 * @param nCols number of columns of the matrix
 * @param shift used to compute the correct position of the file where the current process has to write
 * @param it the iteration number, used to name the output file
 * @param firstRow the first row to save (null for all processes except the first one)
 * @param lastRow the last row to save (null for all processes except the last one)
 */
void save_gnuplot( double *M, size_t nRows, size_t nCols, uint shift, size_t it, double* firstRow, double* lastRow);
