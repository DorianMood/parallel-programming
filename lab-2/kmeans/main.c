#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/types.h>
#include <fcntl.h>
#include <omp.h>
#include "getopt.h"

#include "kmeans.h"

extern double wtime(void);

int num_omp_threads = 0;

/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0) {
#ifdef PARALLEL
    char *help =
        "Usage: %s -s <number objects> -k <number clusters> -n <number threads> [ -t <threshold> ]\n";
#else
    char *help =
        "Usage: %s -s <number objects> -k <number clusters> [ -t <threshold> ]\n";
#endif
    fprintf(stderr, help, argv0);
    exit(-1);
}

int main(int argc, char **argv) {
           int     opt;
    extern char   *optarg;
    extern int     optind;
           int     nclusters = 0;
           float  *buf;
           float **attributes;
           float **cluster_centres=NULL;
           int     i, j;    
           int     numAttributes = 24;
           int     numObjects = 0;        
           char    line[1024];           
           float   threshold = 0.001;
		   double  timing;		   

#ifdef PARALLEL
	while ( (opt=getopt(argc,argv,"s:k:t:n:?"))!= EOF) {
		switch (opt) {
            case 's': numObjects = atoi(optarg);
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'k': nclusters = atoi(optarg);
                      break;
			case 'n': num_omp_threads = atoi(optarg);
					  break;
            case '?': usage(argv[0]);
                      break;
            default: usage(argv[0]);
                      break;
        }
    }
    if ((numObjects == 0) || (nclusters == 0) || (num_omp_threads == 0))
        usage(argv[0]);    

#else
	while ( (opt=getopt(argc,argv,"s:k:t:?"))!= EOF) {
        switch (opt) {
            case 's': numObjects = atoi(optarg);
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'k': nclusters = atoi(optarg);
                      break;
            case '?': usage(argv[0]);
                      break;
            default: usage(argv[0]);
                      break;
        }
    }
    if ((numObjects == 0) || (nclusters == 0))
        usage(argv[0]);    
#endif

    /* allocate space for attributes[] and read attributes of all objects */
    buf           = (float*) malloc(numObjects*numAttributes*sizeof(float));
    attributes    = (float**)malloc(numObjects*             sizeof(float*));
    attributes[0] = (float*) malloc(numObjects*numAttributes*sizeof(float));

    srand(1234);   

    for (i=1; i<numObjects; i++)
        attributes[i] = attributes[i-1] + numAttributes;

    for (i=0; i<numObjects*numAttributes; i++)
        buf[i] = (int)rand() % 16;

	memcpy(attributes[0], buf, numObjects*numAttributes*sizeof(float));

	timing = omp_get_wtime();

    cluster_centres = NULL;

    cluster(numObjects,
            numAttributes,
            attributes,           /* [numObjects][numAttributes] */                
            nclusters,
            threshold,
            &cluster_centres   
           );

    timing = omp_get_wtime() - timing;
	
	printf("number of Clusters: %d\n", nclusters); 
    printf("number of Objects: %d\n", numObjects);
	printf("number of Attributes per Object: %d\n\n",numAttributes); 

/*  	printf("Cluster Centers Output\n"); 
	printf("The first number is cluster number and the following data is arribute value\n");
	printf("=============================================================================\n\n");
	
    for (i=0; i< nclusters; i++) {
		printf("%d: ", i);
        for (j=0; j<numAttributes; j++)
            printf("%.2f ", cluster_centres[i][j]);
        printf("\n\n");
    }
*/

	printf("Time for process: %f s\n", timing);

    free(attributes);
    free(cluster_centres[0]);
    free(cluster_centres);
    free(buf);
    return(0);
}

