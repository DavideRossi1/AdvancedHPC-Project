#pragma once
#include <mpi.h>

struct Timings{
    double initTime;
    double evolveTime;
    double saveTime;
    double totalTime;

    double programStart;
    double start;
};

void printTimings(struct Timings* t, int myRank, int NPEs);

inline void start(struct Timings* t){
    MPI_Barrier(MPI_COMM_WORLD);
    t->start = MPI_Wtime();
}

inline double end(struct Timings* t){
    MPI_Barrier(MPI_COMM_WORLD);
    return MPI_Wtime()- t->start;
}

double seconds();