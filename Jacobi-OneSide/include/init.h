/**
 * @file init.h
 * @author Davide Rossi
 * @brief Header file for the init function
 * @date 2024-06
 * 
 */
#pragma once
#include <stdlib.h>
#include <timer.h>

/**
 * @brief Initialize a matrix for the Jacobi method
 * 
 * @param matrix matrix with the current values
 * @param matrix_new matrix with the zero values
 * @param nRows number of rows of the matrix
 * @param nCols number of columns of the matrix
 * @param prev previous MPI process rank (MPI_PROC_NULL if not present)
 * @param next next MPI process rank (MPI_PROC_NULL if not present)
 * @param shift shift used to correctly fill the first column
 */
void init(double* matrix, double* matrix_new, size_t nRows, size_t nCols, double* firstRow, double* lastRow, uint shift, uint myRank, uint NPEs);
