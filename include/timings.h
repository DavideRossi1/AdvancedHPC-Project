
struct Timings{
    double initTime;
    double initCommTime;
    double gatherTime;
    double resAllocTime;
    double dgemmTime;
    double placeTime;
    double multTime;
    double totalTime;
    double programStart;
    double start;
};

void printTimings(struct Timings* t, int myRank, int NPEs);