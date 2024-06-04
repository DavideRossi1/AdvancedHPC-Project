#pragma once
#include <mpi.h>

// Timer struct, contains the results of the different phases of the program
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

// Print the timings of the different phases of the program
void printTimings(struct Timer* t, int myRank, int NPEs);

// Start the timer
inline void start(struct Timer* t){
    MPI_Barrier(MPI_COMM_WORLD);
    t->start = MPI_Wtime();
}

// End the timer and return the time elapsed from the start
inline double end(struct Timer* t){
    MPI_Barrier(MPI_COMM_WORLD);
    return MPI_Wtime()- t->start;
}

// double seconds();