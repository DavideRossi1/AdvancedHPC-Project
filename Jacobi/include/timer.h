#pragma once
#include <mpi.h>

struct Timer{
    double initACC;
    double copyin;
    double init;
    double update;
    double sendRecv;
    double evolve;
    double save;
    double copyout;
    double total;

    double programStart;
    double evolveStart;
    double start;
};

void printTimings(struct Timer* t, int myRank, int NPEs);

inline void start(struct Timer* t){
    MPI_Barrier(MPI_COMM_WORLD);
    t->start = MPI_Wtime();
}

inline double end(struct Timer* t){
    MPI_Barrier(MPI_COMM_WORLD);
    return MPI_Wtime()- t->start;
}

// double seconds();