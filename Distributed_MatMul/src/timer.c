#include <stdio.h>

#include "timer.h"

void printTimings(struct Timer* t, int myRank, int NPEs)
{
    struct Timer maxT;
    struct Timer avgT;
    MPI_Reduce(&t->initCuda, &maxT.initCuda, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->init,     &maxT.init,     1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->initComm, &maxT.initComm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->gather,   &maxT.gather,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->resAlloc, &maxT.resAlloc, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->dgemm,    &maxT.dgemm,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->place,    &maxT.place,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->mult,     &maxT.mult,     1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->total,    &maxT.total,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&t->initCuda, &avgT.initCuda, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->init,     &avgT.init,     1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->initComm, &avgT.initComm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->gather,   &avgT.gather,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->resAlloc, &avgT.resAlloc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->dgemm,    &avgT.dgemm,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->place,    &avgT.place,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->mult,     &avgT.mult,     1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->total,    &avgT.total,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(!myRank) {
        printf("%f;%f;%f;%f;%f;%f;%f;%f;%f\n", maxT.initCuda, maxT.init, maxT.initComm, maxT.gather, maxT.resAlloc, maxT.dgemm, maxT.place, maxT.mult, maxT.total);
        printf("%f;%f;%f;%f;%f;%f;%f;%f;%f\n", avgT.initCuda/NPEs, avgT.init/NPEs, avgT.initComm/NPEs, avgT.gather/NPEs, avgT.resAlloc/NPEs, avgT.dgemm/NPEs, avgT.place/NPEs, avgT.mult/NPEs, avgT.total/NPEs);
    }
}
