#include <sys/time.h>
#include <stdio.h>

#include "timer.h"

void printTimings(struct Timer* t, int myRank, int NPEs)
{
    struct Timer maxT;
    struct Timer avgT;
    MPI_Reduce(&t->init,     &maxT.init,     1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->update,   &maxT.update,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->sendRecv, &maxT.sendRecv, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->evolve,   &maxT.evolve,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->save,     &maxT.save,     1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->total,    &maxT.total,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&t->init,     &avgT.init,     1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->update,   &avgT.update,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->sendRecv, &avgT.sendRecv, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->evolve,   &avgT.evolve,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->save,     &avgT.save,     1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->total,    &avgT.total,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(!myRank) {
        printf("%f;%f;%f;%f;%f;%f\n", maxT.init, maxT.update, maxT.sendRecv, maxT.evolve, maxT.save, maxT.total);
        printf("%f;%f;%f;%f;%f;%f\n", avgT.init/NPEs, avgT.update/NPEs, avgT.sendRecv/NPEs, avgT.evolve/NPEs, avgT.save/NPEs, avgT.total/NPEs);
    }
}
