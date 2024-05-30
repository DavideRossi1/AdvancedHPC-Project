#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "print.h"


void printMatrixThrSafe(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs) 
{
  MPI_Barrier(MPI_COMM_WORLD);
  #pragma acc update self(matrix[:nRows*nCols])
  if (myRank) {
    if (myRank == NPEs-1) nRows++;
    MPI_Send(&matrix[nCols], (nRows-2)*nCols, MPI_DOUBLE, 0, myRank, MPI_COMM_WORLD);
  } else 
  {
    size_t offset = 0;
    size_t charRowSize = nCols*7; // 7 is the number of characters in the format string ('90.000\t')
    // Note: we actually also have a 100.000\t, but it will be a single value in the matrix, so the space needed 
    // for it is balanced by the zeroes in the first row. We'll overall allocate slightly more memory than needed,
    // but it's ok since we'll only use this function for debugging purposes.
    size_t entireCharMatrixSize = nCols * charRowSize; // the entire matrix is nCols*nCols big
    char *entireCharMatrix = (char*)malloc(entireCharMatrixSize * sizeof(char));
    // skip the last row if process 0 is not the last one
    uint nRowsCurrentProc = nRows - (NPEs > 1 ? 1 : 0);
    char* charMatrix = (char*)malloc(nRowsCurrentProc * charRowSize * sizeof(char));
    convertMatrix(matrix, charMatrix, nRowsCurrentProc, nCols);
    offset += snprintf(entireCharMatrix + offset, entireCharMatrixSize - offset, "%s", charMatrix);
    free(charMatrix);
    for (uint i = 1; i < NPEs; i++) 
    {
      nRowsCurrentProc = (nCols-2)/NPEs + (i < (nCols-2)%NPEs ? 1 : 0) + (i == NPEs-1 ? 1 : 0);
      charMatrix = (char*)malloc(nRowsCurrentProc * charRowSize * sizeof(char));
      double *buf = (double *)malloc(nRowsCurrentProc * nCols * sizeof(double));
      MPI_Recv(buf, nRowsCurrentProc * nCols, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      convertMatrix(buf, charMatrix, nRowsCurrentProc, nCols);
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
  int offset = 0;
  const size_t charMatrixSize = nRows*(nCols*7);
  for (uint i = 0; i < nRows; i++) {  
    for (uint j = 0; j < nCols; j++){
      char* format = j < nCols-1 ? "%.3f\t" : "%.3f\n";
      offset += snprintf(charMatrix+offset, charMatrixSize-offset, format, matrix[i * nCols + j]);
    }
  }
}


void printMatrixDistributed(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs) 
{
  #pragma acc update self(matrix[:nRows*nCols])
  if (myRank) {
    if (myRank == NPEs-1) nRows++;
    MPI_Send(&matrix[nCols], (nRows-2) * nCols, MPI_DOUBLE, 0, myRank, MPI_COMM_WORLD);
  } else 
  {
    uint nRowsCurrentProc = nRows - (NPEs > 1 ? 1 : 0);
    printMatrix(matrix, nRowsCurrentProc, nCols);
    for (uint i = 1; i < NPEs; i++) 
    {
      nRowsCurrentProc = (nCols-2)/NPEs + (i < (nCols-2)%NPEs ? 1 : 0) + (i == NPEs-1 ? 1 : 0);
      double *buf = (double *)malloc(nRowsCurrentProc * nCols * sizeof(double));
      MPI_Recv(buf, nRowsCurrentProc * nCols, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printMatrix(buf, nRowsCurrentProc, nCols);
      free(buf);
    }
    printf("\n");
  }
}

void printMatrix(double *matrix, uint nRows, uint nCols) {
  for (uint i = 0; i < nRows; i++) {
    for (uint j = 0; j < nCols; j++)
      printf("%.3f\t", matrix[i*nCols + j]);
    printf("\n");
  }
}

void save_gnuplot( double *M, size_t dim, uint firstRow, uint lastRow, uint shift)
{
  const double h = 0.1;
  MPI_File file;
  MPI_File_open(MPI_COMM_WORLD, "solution.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
  MPI_Offset of = shift*dim*3*sizeof(double);
  for (size_t i = firstRow; i < lastRow; i++){
    for (size_t j = 0; j < dim; j++){
      double buffer[3] = {h*j, -h*(shift+i), M[i*dim+j]};
      MPI_File_write_at(file, of + (i*dim+j)*3*sizeof(double), buffer, 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
  }
  MPI_File_close(&file);
}
