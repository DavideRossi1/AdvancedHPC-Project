#include <mpi.h>
#include <omp.h>

#include "init.h"

void init(double* matrix, double* matrix_new, size_t nRows, size_t nCols, int prev, int next, uint shift, struct Timer* t)
{
  start(t);
#ifdef _OPENACC
#pragma acc parallel loop present(matrix[:nRows*nCols], matrix_new[:nRows*nCols])
#else
#pragma omp parallel for
#endif
  for(size_t k = 0; k< nRows*nCols; k++){ 
    matrix[k] = 0.0;
    matrix_new[k] = 0.0;
  }
  //fill initial values  
  //fill the inner square
#ifdef _OPENACC
#pragma acc parallel loop collapse(2) present(matrix[:nRows*nCols])
#else
#pragma omp parallel for collapse(2)
#endif
  for(size_t i = 1; i < nRows-1; ++i )
    for(size_t j = 1; j < nCols-1; ++j )
      matrix[ i*nCols + j ] = 0.5;   
  // set up borders 
  double increment = 100.0/(nCols-1);
  // fill the first column
#ifdef _OPENACC
#pragma acc parallel loop present(matrix[:nRows*nCols], matrix_new[:nRows*nCols])
#else
#pragma omp parallel for
#endif
  for(size_t i = 1; i < nRows-1; ++i ){
    matrix[ i*nCols ] = (i+shift)*increment;
    matrix_new[ i*nCols ] = (i+shift)*increment;
  }
  // fill the last row
  if (next == MPI_PROC_NULL){
#ifdef _OPENACC
#pragma acc parallel loop present(matrix[:nRows*nCols], matrix_new[:nRows*nCols])
#else
#pragma omp parallel for
#endif
    for(size_t i = 1; i < nCols; ++i ){
      matrix[ nRows*nCols-1-i ] = i*increment;
      matrix_new[ nRows*nCols-1-i ] = i*increment;
    }
  }
#pragma acc wait
  t->init = end(t);
  start(t);
  MPI_Request send_request[4], recv_request[4];
#pragma acc host_data use_device(matrix, matrix_new)
  {
    MPI_Isend(&matrix[nCols], nCols, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &send_request[0]);
    MPI_Irecv(&matrix[0], nCols, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &recv_request[0]);

    MPI_Isend(&matrix_new[nCols], nCols, MPI_DOUBLE, prev, 2, MPI_COMM_WORLD, &send_request[1]);
    MPI_Irecv(&matrix_new[0], nCols, MPI_DOUBLE, prev, 3, MPI_COMM_WORLD, &recv_request[1]);

    MPI_Isend(&matrix[(nRows - 2) * nCols], nCols, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &send_request[2]);
    MPI_Irecv(&matrix[(nRows - 1) * nCols], nCols, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &recv_request[2]);

    MPI_Isend(&matrix_new[(nRows - 2) * nCols], nCols, MPI_DOUBLE, next, 3, MPI_COMM_WORLD, &send_request[3]);
    MPI_Irecv(&matrix_new[(nRows - 1) * nCols], nCols, MPI_DOUBLE, next, 2, MPI_COMM_WORLD, &recv_request[3]);
  }
  MPI_Waitall(4, send_request, MPI_STATUSES_IGNORE);
  MPI_Waitall(4, recv_request, MPI_STATUSES_IGNORE);
  t->sendRecv += end(t);  
}
