#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#include "print.h"


size_t getRowSize(uint matrixSize) {
  size_t maxVal = matrixSize*matrixSize;
  // Explanation: 
  // - if the matrix has more than 100 elements, the maximum value will have 3 digits, so we need 8 characters to represent it (123.456\t)
  // - if the matrix has more than 10 elements, the maximum value will have 2 digits, so we need 7 characters to represent it (12.345\t)
  // - if the matrix has 10 or less elements, the maximum value will have 1 digit, so we need 6 characters to represent it (1.234\t)
  // This is done just to save some memory, we could have simply used 8 characters for all cases, or even more to cover any kind of matrix,
  // but that would be a waste of memory.
  int elemSize = maxVal > 100 ? 8 : maxVal > 10 ? 7 : 6;
  return matrixSize * elemSize;
}

void convertRow(double *row, char* rowChar, uint nCols)
{
  size_t offset = 0;
  const size_t charMatrixSize = getRowSize(nCols) + 1; 
  for (uint j = 0; j < nCols; j++){
    char* format = j < nCols-1 ? "%.3f\t" : "%.3f\n";
    offset += snprintf(rowChar+offset, charMatrixSize-offset, format, row[j]);
  }
}


void printMatrixThrSafe(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs, double* firstRow, MPI_Win lastRowWin) 
{
  MPI_Barrier(MPI_COMM_WORLD);
  if (myRank) {
    MPI_Send(matrix, nRows*nCols, MPI_DOUBLE, 0, myRank, MPI_COMM_WORLD);
  } else 
  {
    size_t offset = 0;
    uint charRowSize = getRowSize(nCols);
    size_t entireCharMatrixSize = nCols * charRowSize + 1; // the entire matrix is nCols*nCols big, +1 for the null terminator
    char *entireCharMatrix = (char*)malloc(entireCharMatrixSize * sizeof(char));
    char* rowChar = (char*)malloc(charRowSize * sizeof(char));
    convertRow(firstRow, rowChar, nCols);
    offset += snprintf(entireCharMatrix + offset, entireCharMatrixSize - offset, "%s", rowChar);
    char* charMatrix = (char*)malloc(nRows * charRowSize * sizeof(char));
    convertMatrix(matrix, charMatrix, nRows, nCols);
    offset += snprintf(entireCharMatrix + offset, entireCharMatrixSize - offset, "%s", charMatrix);
    double *buf = (double *)malloc(nRows * nCols * sizeof(double));
    for (uint i = 1; i < NPEs; i++) 
    {
      uint nRowsCurrentProc = (nCols-2)/NPEs + (i < (nCols-2)%NPEs ? 1 : 0);
      MPI_Recv(buf, nRowsCurrentProc * nCols, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      convertMatrix(buf, charMatrix, nRowsCurrentProc, nCols);
      offset += snprintf(entireCharMatrix + offset, entireCharMatrixSize - offset, "%s", charMatrix);
    }
    double* lastRowBuf = (double*)malloc(nCols * sizeof(double));
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, NPEs-1, 0, lastRowWin);
    MPI_Get(lastRowBuf, nCols, MPI_DOUBLE, NPEs-1, 0, nCols, MPI_DOUBLE, lastRowWin);
    MPI_Win_unlock(NPEs-1, lastRowWin);
    convertRow(lastRowBuf, rowChar, nCols);
    offset += snprintf(entireCharMatrix + offset, entireCharMatrixSize - offset, "%s", rowChar);
    printf("%s\n", entireCharMatrix);
    free(buf);
    free(lastRowBuf);
    free(charMatrix);
    free(rowChar);
    free(entireCharMatrix);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}


void convertMatrix(double *matrix, char* charMatrix, uint nRows, uint nCols) {
  // build a string that contains the matrix
  size_t offset = 0;
  const size_t charMatrixSize = nRows*getRowSize(nCols) + 1; 
  for (uint i = 0; i < nRows; i++) {  
    for (uint j = 0; j < nCols; j++){
      char* format = j < nCols-1 ? "%.3f\t" : "%.3f\n";
      offset += snprintf(charMatrix+offset, charMatrixSize-offset, format, matrix[i * nCols + j]);
    }
  }
}

void printRow(double *row, uint nCols) {
  for (uint i = 0; i < nCols; i++)
    printf("%.3f\t", row[i]);
  printf("\n");
}


void printMatrixDistributed(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs, double* firstRow, MPI_Win lastRowWin) 
{
  if (myRank) {
    MPI_Send(matrix, nRows * nCols, MPI_DOUBLE, 0, myRank, MPI_COMM_WORLD);
  } else 
  {
    printRow(firstRow, nCols);
    printMatrix(matrix, nRows, nCols);
    double *buf = (double *)malloc(nRows * nCols * sizeof(double)); // first process for sure has more rows than any other proc
    for (uint i = 1; i < NPEs; i++) 
    {
      uint nRowsCurrentProc = (nCols-2)/NPEs + (i < (nCols-2)%NPEs ? 1 : 0);
      MPI_Recv(buf, nRowsCurrentProc * nCols, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printMatrix(buf, nRowsCurrentProc, nCols); 
    }
    double* lastRowBuf = (double*)malloc(nCols * sizeof(double));
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, NPEs-1, 0, lastRowWin);
    MPI_Get(lastRowBuf, nCols, MPI_DOUBLE, NPEs-1, 0, nCols, MPI_DOUBLE, lastRowWin);
    MPI_Win_unlock(NPEs-1, lastRowWin);
    printRow(lastRowBuf, nCols);
    free(buf);
    free(lastRowBuf);
    printf("\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void printMatrix(double *matrix, uint nRows, uint nCols) {
  for (uint i = 0; i < nRows; i++)
    printRow(matrix + i*nCols, nCols);
}

void save_gnuplot( double *M, size_t nRows, size_t nCols, uint shift, size_t it, double* firstRow, double* lastRow)
{
  const double h = 0.1;
  MPI_File file;
  char* filename = (char*)malloc(25*sizeof(char));
  sprintf(filename, "output/solution%zu.dat", it);
  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
  if (firstRow != NULL){
#pragma omp parallel for
    for (size_t j = 0; j < nCols; j++){
      double buffer[3] = {h*j, -h*shift, firstRow[j]};
      MPI_File_write_at(file, j*3*sizeof(double), buffer, 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
  }
  MPI_Offset of = shift*nCols*3*sizeof(double);
#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < nRows; i++){
    for (size_t j = 0; j < nCols; j++){
      double buffer[3] = {h*j, -h*(shift+i+1), M[i*nCols+j]};
      MPI_File_write_at(file, of + ((i+1)*nCols+j)*3*sizeof(double), buffer, 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
  }
  if (lastRow != NULL){
#pragma omp parallel for
    for (size_t j = 0; j < nCols; j++){
      double buffer[3] = {h*j, -h*(shift+nRows+1), lastRow[j]};
      MPI_File_write_at(file, of + ((nRows+1)*nCols+j)*3*sizeof(double), buffer, 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
  }
  MPI_File_close(&file);
}
