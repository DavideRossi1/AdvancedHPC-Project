
struct Timings{
    double initTime;
    double evolveTime;
    double saveTime;
    double totalTime;

    double programStart;
    double start;
};

void printTimings(struct Timings* t, int myRank, int NPEs);

double seconds();