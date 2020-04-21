#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>

#include "kmeans.h"

extern double wtime(void);
extern int num_omp_threads;

float euclid_dist_2(float *a, float *b, int ndimention)
{
    float summa = 0.0;
    for (int i = 0; i < ndimention; i++)
    {
        summa += pow(a[i] - b[i], 2);
    }
    return sqrt(summa);
}

/*----< serial_clustering() >---------------------------------------------*/
float **serial_clustering(float **feature, /* in: [npoints][nfeatures] */
                          int nfeatures,
                          int npoints,
                          int nclusters,
                          float threshold,
                          int *membership) /* out: [npoints] */
{

    float **clusters; /* out: [nclusters][nfeatures] */
    /* allocate space for returning variable clusters[] */
    clusters = (float **)malloc(nclusters * sizeof(float *));
    clusters[0] = (float *)malloc(nclusters * nfeatures * sizeof(float));
    for (int i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    /* randomly pick cluster centers */
    int n = 0;
    for (int i = 0; i < nclusters; i++)
    {
        //n = (int)rand() % npoints;
        for (int j = 0; j < nfeatures; j++)
            clusters[i][j] = feature[n][j];
        n++;
    }

    // PUT YOUR CODE HERE
    for (int i = 0; i < npoints; i++)
    {
        float min_distance = euclid_dist_2(feature[i], clusters[0], nfeatures);
        membership[i] = clusters[0];
        for (int j = 1; j < nclusters; j++)
        {
            // Look for minimal distance d (feature[i], clusters[j])
            float distance = euclid_dist_2(feature[i], clusters[j], nfeatures);
            if (min_distance > distance)
            {
                min_distance = distance;
                membership[i] = clusters[j];
            }
        }
    }

    return clusters;
}

/*----< parallel_clustering() >---------------------------------------------*/
float **parallel_clustering(float **feature, /* in: [npoints][nfeatures] */
                            int nfeatures,
                            int npoints,
                            int nclusters,
                            float threshold,
                            int *membership) /* out: [npoints] */
{

    float **clusters; /* out: [nclusters][nfeatures] */
    int nthreads;

    nthreads = num_omp_threads;

    /* allocate space for returning variable clusters[] */
    clusters = (float **)malloc(nclusters * sizeof(float *));
    clusters[0] = (float *)malloc(nclusters * nfeatures * sizeof(float));
    for (int i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    /* randomly pick cluster centers */
    int n = 0;
    for (int i = 0; i < nclusters; i++)
    {
        //n = (int)rand() % npoints;
        for (int j = 0; j < nfeatures; j++)
            clusters[i][j] = feature[n][j];
        n++;
    }

    // Public variables
    int i, j;
    float min_distance = FLT_MAX;
    #pragma omp parallel for private(min_distance)
    for (i = 0; i < npoints; i++)
    {
        for (j = 0; j < nclusters; j++)
        {
            // Look for minimal distance d (feature[i], clusters[j])
            float distance = euclid_dist_2(feature[i], clusters[j], nfeatures);
            if (min_distance > distance)
            {
                min_distance = distance;
                membership[i] = clusters[j];
            }
        }
        min_distance = FLT_MAX;
    }

    return clusters;
}
