/**
 * @file evolve.h
 * @author Davide Rossi
 * @brief Header file for the evolve function
 * @date 2024-06
 * 
 */
#pragma once
#include <stdlib.h>

/**
 * @brief Evolve the matrix using the Jacobi method
 * 
 * @param matrix matrix with the current values
 * @param matrix_new matrix that will contain the new values
 * @param nRows number of rows of the matrix
 * @param nCols number of columns of the matrix
 * @param firstRow first row of the matrix
 * @param lastRow last row of the matrix
 */
void evolve( double* matrix, double* matrix_new, size_t nRows, size_t nCols, double* firstRow, double* lastRow);
