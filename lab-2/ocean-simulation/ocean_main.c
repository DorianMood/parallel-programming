#include <stdio.h>
#include <stdlib.h>

#include "getopt.h"
#include "hwtimer.h"

/* Implement this function in serial_ocean and omp_ocean */
#ifdef PARALLEL
extern void ocean (int** grid, int dim, int timesteps, int threads);
#else
extern void ocean (int** grid, int dim, int timesteps);
#endif

void usage(char *argv0) {
#ifdef PARALLEL
    char *help =
        "Usage: %s -d <dimension> -t <timesteps> -n <threads>\n";
#else
    char *help =
        "Usage: %s -d <dimension> -t <timesteps>\n";
#endif
    fprintf(stderr, help, argv0);
    exit(-1);
}

void printGrid(int** grid, int dim);

int main(int argc, char* argv[])
{
    int opt;
    extern char *optarg;
    int dim=0, timesteps=0, threads=0;
    int** grid;
    int i,j,t;

    hwtimer_t timer;
    initTimer(&timer);
    
    /********************Get the arguments correctly (start) **************************/
    /* 
    Three input Arguments to the program
    1. X Dimension of the grid
    2. Y dimension of the grid
    3. number of timesteps the algorithm is to be performed
    */
#ifdef PARALLEL
	while ( (opt=getopt(argc,argv,"d:t:n:?"))!= EOF) {
		switch (opt) {
            case 'd': dim = atoi(optarg);
                      break;
            case 't': timesteps=atoi(optarg);
                      break;
            case 'n': threads = atoi(optarg);
                      break;
            case '?': usage(argv[0]);
                      break;
            default: usage(argv[0]);
                      break;
        }
    }
    if ((dim == 0) || (timesteps == 0) || (threads == 0))
        usage(argv[0]);
#else
	while ( (opt=getopt(argc,argv,"d:t:?"))!= EOF) {
		switch (opt) {
            case 'd': dim = atoi(optarg);
                      break;
            case 't': timesteps=atoi(optarg);
                      break;
            case '?': usage(argv[0]);
                      break;
            default: usage(argv[0]);
                      break;
        }
    }
    if ((dim == 0) || (timesteps == 0))
        usage(argv[0]);
#endif
    ///////////////////////Get the arguments correctly (end) //////////////////////////


    /*********************create the grid as required (start) ************************/
    /*
    The grid needs to be allocated as per the input arguments and randomly initialized.
    Remember during allocation that we want to gaurentee a contiguous block, hence the
    nasty pointer math.

    To test your code for correctness please comment this section of random initialization.
    */
    srand(1234);
    grid = (int**) malloc(dim*sizeof(int*));
    int *temp = (int*) malloc(dim*dim*sizeof(int));
    for (i=0; i<dim; i++) {
        grid[i] = &temp[i*dim];
    }
    for (i=0; i<dim; i++) {
        for (j=0; j<dim; j++) {
            grid[i][j] = rand();
        }
    }
    ///////////////////////create the grid as required (end) //////////////////////////

    startTimer(&timer); // Start the time measurment here before the algorithm starts

#ifdef PARALLEL
    ocean(grid, dim, timesteps, threads);
#else
    ocean(grid, dim, timesteps);
#endif

    stopTimer(&timer); // End the time measuremnt here since the algorithm ended

    //Do the time calcuclation
    printf("Total Execution time: %lld ns\n", getTimerNs(&timer));

    // Free the memory we allocated for grid
    free(temp);
    free(grid);

    return EXIT_SUCCESS;
}
