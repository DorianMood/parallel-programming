#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#include "kmeans.h"

/*---< cluster() >-----------------------------------------------------------*/
int cluster(int      numObjects,      /* number of input objects */
            int      numAttributes,   /* size of attribute of each object */
            float  **attributes,      /* [numObjects][numAttributes] */            
            int      nclusters,
            float    threshold,       /* in:   */
            float ***cluster_centres /* out: [best_nclusters][numAttributes] */
    
            )
{
    int    *membership;
    float **tmp_cluster_centres;

    membership = (int*) malloc(numObjects * sizeof(int));

	/* perform regular Kmeans */
#ifdef PARALLEL
    tmp_cluster_centres = parallel_clustering(attributes,
                                            numAttributes,
                                            numObjects,
                                            nclusters,
                                            threshold,
                                            membership);      
#else
    tmp_cluster_centres = serial_clustering(attributes,
                                            numAttributes,
                                            numObjects,
                                            nclusters,
                                            threshold,
                                            membership);      
#endif

    if (*cluster_centres) {
		free((*cluster_centres)[0]);
        free(*cluster_centres);
    }
    *cluster_centres = tmp_cluster_centres;

   
	free(membership);

    return 0;
}

