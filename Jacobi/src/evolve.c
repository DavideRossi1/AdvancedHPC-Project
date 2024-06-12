#include <mpi.h>
#include <omp.h>

#include "evolve.h"

void evolve( double* matrix, double* matrix_new, size_t nRows, size_t nCols, int prev, int next, struct Timer* t)
{
  //This will be a row dominant program.
  start(t);
#ifdef _OPENACC
#pragma acc parallel loop collapse(2) present(matrix[:nRows*nCols], matrix_new[:nRows*nCols])
#else
#pragma omp parallel for collapse(2)
#endif
  for(size_t i = 1; i < nRows-1; ++i )
    for(size_t j = 1; j < nCols-1; ++j ) {
      size_t currentEl = i*nCols + j;
      matrix_new[currentEl] = 0.25*( matrix[currentEl-nCols] + matrix[currentEl+1] + 
                                     matrix[currentEl+nCols] + matrix[currentEl-1] );
    }
#pragma acc wait
  t->update += end(t);
  start(t);
  MPI_Request send_request[2], recv_request[2];
#pragma acc host_data use_device(matrix, matrix_new)
  {
    MPI_Isend(&matrix_new[nCols], nCols, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &send_request[0]);
    MPI_Irecv(&matrix_new[0], nCols, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &recv_request[0]);

    MPI_Isend(&matrix_new[(nRows - 2) * nCols], nCols, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &send_request[1]);
    MPI_Irecv(&matrix_new[(nRows - 1) * nCols], nCols, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &recv_request[1]);
  }
  MPI_Waitall(2, send_request, MPI_STATUSES_IGNORE);
  MPI_Waitall(2, recv_request, MPI_STATUSES_IGNORE);
  t->sendRecv += end(t);  
}