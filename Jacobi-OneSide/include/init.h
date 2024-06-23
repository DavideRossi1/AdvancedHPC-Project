/**
 * @file init.h
 * @author Davide Rossi
 * @brief Header file for the init function
 * @date 2024-06
 * 
 */
#pragma once
#include <stdlib.h>

/**
 * @brief Initialize a matrix for the Jacobi method
 * 
 * @param matrix matrix with the current values
 * @param matrix_new matrix with the zero values
 * @param nRows number of rows of the matrix
 * @param nCols number of columns of the matrix
 * @param firstRow first row of the matrix
 * @param lastRow last row of the matrix
 * @param shift shift for the current process
 * @param myRank process rank
 * @param NPEs total number of processes
 */
void init(double* matrix, double* matrix_new, size_t nRows, size_t nCols, double* firstRow, double* lastRow, uint shift, uint myRank, uint NPEs);
