/**
 * @file print.c
 * @author Davide Rossi
 * @brief Source file for the printing functions
 * @date 2024-06
 * 
 */
#include <stdio.h>
#include <mpi.h>

#include "print.h"

void printMatrixThrSafe(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs) 
{
  MPI_Barrier(MPI_COMM_WORLD);
  if (myRank) {
    MPI_Send(matrix, nRows * nCols, MPI_DOUBLE, 0, myRank, MPI_COMM_WORLD);
  } else 
  {
    size_t offset = 0;
    uint charRowSize = nCols*8;  // suppose each element to have at most 7 chars plus the tab (ddd.ddd\t)
    size_t entireCharMatrixSize = nCols * charRowSize + 1; // the entire matrix is nCols*nCols big, +1 for the null terminator
    char *entireCharMatrix = (char*)malloc(entireCharMatrixSize * sizeof(char));
    char* charMatrix = (char*)malloc(nRows * charRowSize * sizeof(char));
    convertMatrix(matrix, charMatrix, nRows, nCols);
    offset += snprintf(entireCharMatrix + offset, entireCharMatrixSize - offset, "%s", charMatrix);
    free(charMatrix);
    for (uint i = 1; i < NPEs; i++) 
    {
      uint nRowsSender = nCols / NPEs + (i < nCols % NPEs ? 1 : 0);
      charMatrix = (char*)malloc((nRowsSender * charRowSize + 1) * sizeof(char)); // +1 for the null terminator
      double *buf = (double *)malloc(nRowsSender * nCols * sizeof(double));
      MPI_Recv(buf, nRowsSender * nCols, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      convertMatrix(buf, charMatrix, nRowsSender, nCols);
      offset += snprintf(entireCharMatrix + offset, entireCharMatrixSize - offset, "%s", charMatrix);
      free(charMatrix);
      free(buf);
    }
    printf("%s\n", entireCharMatrix);
    free(entireCharMatrix);
  }
}

void convertMatrix(double *matrix, char* charMatrix, uint nRows, uint nCols) {
  // build a string that contains the matrix
  size_t offset = 0;
  const size_t charMatrixSize = nRows*nCols*8 + 1; 
  for (uint i = 0; i < nRows; i++) {  
    for (uint j = 0; j < nCols; j++){
      char* format = j < nCols-1 ? "%.3f\t" : "%.3f\n";
      offset += snprintf(charMatrix+offset, charMatrixSize-offset, format, matrix[i * nCols + j]);
    }
  }
}

void printMatrixDistributed(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs) 
{
  if (myRank) {
    MPI_Send(matrix, nRows * nCols, MPI_DOUBLE, 0, myRank, MPI_COMM_WORLD);
  } 
  else {
    printMatrix(matrix, nRows, nCols);
    for (uint i = 1; i < NPEs; i++) {
      uint nRowsSender = nCols / NPEs + (i < nCols % NPEs ? 1 : 0);
      double *buf = (double *)malloc(nRowsSender * nCols * sizeof(double));
      MPI_Recv(buf, nRowsSender * nCols, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printMatrix(buf, nRowsSender, nCols);
      free(buf);
    }
    printf("\n");
  }
}

void printMatrix(double *matrix, uint nRows, uint nCols) {
  for (uint i = 0; i < nRows; i++) {
    for (uint j = 0; j < nCols; j++)
      printf("%.3f\t", matrix[i * nCols + j]);
    printf("\n");
  }
}
