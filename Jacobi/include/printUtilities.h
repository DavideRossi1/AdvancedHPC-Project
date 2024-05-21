#pragma once
#include <stdlib.h>
#include <mpi.h>

void printMatrix(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs);

void printMatrixDistributed(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs);

void save_gnuplot( double *M, size_t dim, uint myRank, uint NPEs, uint myWorkSize);

void convertBinToTxt();