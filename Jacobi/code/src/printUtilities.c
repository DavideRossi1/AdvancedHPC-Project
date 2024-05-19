#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "../include/printUtilities.h"

void printMatrix(double *matrix, uint nRows, uint nCols) {
  #ifdef DEBUG
    const char* format = "%.0f ";
  #else
    const char* format = "%.2f ";
  #endif
  size_t bufferSize = nRows*(nCols*7 + 2); // 7 is the number of characters in the format string ('100.00 '), 2 is for the newline and null terminator
  char* buffer = (char*)malloc(bufferSize*sizeof(char));
  // build a string that contains the matrix
  int offset = 0;
  for (uint i = 1; i < nRows-1; i++) {  // skip the first and last rows, which are used to exchange messages with the other procs
    for (uint j = 0; j < nCols; j++){
      offset += snprintf(buffer+offset, bufferSize-offset, format, matrix[i * nCols + j]);
    }
    offset += snprintf(buffer+offset, bufferSize-offset, "\n");
  }
  printf("%s", buffer);
  free(buffer);
}

void printMatrixDistributed(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs) 
{
  if (myRank) {
    MPI_Send(matrix, nRows * nCols, MPI_DOUBLE, 0, myRank, MPI_COMM_WORLD);
  } 
  else {
    double *buf = (double *)malloc(nRows * nCols * sizeof(double));
    printMatrix(matrix, nRows, nCols);
    for (uint i = 1; i < NPEs; i++) {
      uint nLocSender = nCols / NPEs + (i < nCols % NPEs ? 3 : 2);
      MPI_Recv(buf, nLocSender * nCols, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printMatrix(buf, nLocSender, nCols);
    }
    free(buf);
  }
}
