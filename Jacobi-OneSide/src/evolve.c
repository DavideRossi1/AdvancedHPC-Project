/**
 * @file evolve.c
 * @author Davide Rossi
 * @brief Source file for the evolve function
 * @date 2024-06
 * 
 */
#include <mpi.h>
#include <omp.h>

#include "evolve.h"

void evolve( double* matrix, double* matrix_new, size_t nRows, size_t nCols, int prev, int next, struct Timer* t)
{
  // Update matrix_new using matrix values
  start(t);
#pragma omp parallel for collapse(2)
  for(size_t i = 1; i < nRows-1; ++i )
    for(size_t j = 1; j < nCols-1; ++j ) {
      size_t currentEl = i*nCols + j;
      matrix_new[currentEl] = 0.25*( matrix[currentEl-nCols] + matrix[currentEl+1] + 
                                     matrix[currentEl+nCols] + matrix[currentEl-1] );
    }
  t->update += end(t);
  // Exchange the ghost rows
  start(t);
  MPI_Request send_request[2], recv_request[2];
  MPI_Isend(&matrix_new[nCols], nCols, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &send_request[0]);
  MPI_Irecv(&matrix_new[0], nCols, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &recv_request[0]);
  MPI_Isend(&matrix_new[(nRows - 2) * nCols], nCols, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &send_request[1]);
  MPI_Irecv(&matrix_new[(nRows - 1) * nCols], nCols, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &recv_request[1]);
  MPI_Waitall(2, send_request, MPI_STATUSES_IGNORE);
  MPI_Waitall(2, recv_request, MPI_STATUSES_IGNORE);
  t->sendRecv += end(t);  
}
