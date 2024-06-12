/**
 * @file evolve.h
 * @author Davide Rossi
 * @brief Header file for the evolve function
 * @date 2024-06
 * 
 */
#pragma once
#include <stdlib.h>
#include "timer.h"

/**
 * @brief Evolve the matrix using the Jacobi method
 * 
 * @param matrix matrix with the current values
 * @param matrix_new matrix that will contain the new values
 * @param nRows number of rows of the matrix
 * @param nCols number of columns of the matrix
 * @param prev previous MPI process rank (MPI_PROC_NULL if not present)
 * @param next next MPI process rank (MPI_PROC_NULL if not present)
 * @param t struct Timer to save the results of the timing
 */
void evolve( double* matrix, double* matrix_new, size_t nRows, size_t nCols, int prev, int next, struct Timer* t);
