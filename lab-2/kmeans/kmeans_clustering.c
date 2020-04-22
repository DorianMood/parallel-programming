#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <string.h>

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

int find_nearest_point(float *point, int ndimension, float **feature, int nfeatures)
{
    int point_index = 0;
    float distance = FLT_MAX;
    float min_distance = FLT_MAX;
    for (int i = 0; i < nfeatures; i++)
    {
        distance = euclid_dist_2(feature[point_index], point, ndimension);
        if (distance < min_distance)
        {
            min_distance = distance;
            point_index = i;
        }
    }
    return point_index;
}

float *average(float **elements, int nelements, int ndimension)
{
    float *point = (float *)malloc(ndimension * sizeof(float));
    for (int i = 0; i < nelements; i++)
        for (int j = 0; j < ndimension; j++)
            point[j] += elements[i][j] / nelements;
    return point;
}

void calculate_centroids(float **clusters, int nclusters,
                         float **feature, int npoints, int nfeatures,
                         int *membership)
{
    float t = feature[1][1];
    // If not clustered generate random centers
    if (membership[0] == -1)
    {
        int n = 0;
        for (int i = 0; i < nclusters; i++)
        {
            // n = (int)rand() % npoints;
            for (int j = 0; j < nfeatures; j++)
            {
                float x = feature[n][j];
                clusters[i][j] = (float) feature[n][j];
            }
            n++;
        }
        return;
    }
    // Reset cluster centers
    memset(clusters, 0, sizeof(clusters[0][0]) * nclusters * nfeatures);

    // Calculate clusters sizes
    int *clusters_size = (int *)malloc(nclusters * sizeof(int));
    memset(clusters_size, 0, nclusters);
    for (int i = 0; i < npoints; i++)
        clusters[membership[i]]++;

    // Calculate new cluster centers
    for (int i = 0; i < npoints; i++)
    {
        for (int j = 0; j < nfeatures; j++)
        {
            clusters[membership[i]][j] +=
                feature[i][j] / (float)npoints;
        }
    }
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
    clusters = (float **)malloc(sizeof(float[nclusters][nfeatures]));
    memset(clusters, 0, sizeof(float[nclusters][nfeatures]));

    /* Initilize membership */
    memset(membership, -1, sizeof(int[npoints]));

    // PUT YOUR CODE HERE
    int count = 0;
    do
    {
        count = 0;
        // Calculate new centers
        calculate_centroids(clusters, nclusters, feature, npoints, nfeatures, membership);
        // Perform clustering
        float min_distance = FLT_MAX;
        // each point
        for (int i = 0; i < npoints; i++)
        {
            float min_distance = FLT_MAX;
            // match each cluster
            for (int j = 0; j < nclusters; j++)
            {
                // Look for minimal distance d (feature[i], clusters[j])
                float distance = euclid_dist_2(feature[i], clusters[j], nfeatures);
                if (distance < min_distance)
                {
                    min_distance = distance;
                    if (membership[i] != j)
                    {
                        printf("%d", count);
                        count++;
                    }
                    membership[i] = j;
                }
            }
        }
        printf("|%f|", ((float)count / (float)npoints));
    } while (((float)count / (float)npoints) > threshold);

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
#pragma omp parallel for num_threads(num_omp_threads) private(min_distance)
    for (i = 0; i < npoints; i++)
    {
        for (j = 0; j < nclusters; j++)
        {
            // Look for minimal distance d (feature[i], clusters[j])
            float distance = euclid_dist_2(feature[i], clusters[j], nfeatures);
            if (min_distance > distance)
            {
                min_distance = distance;
                membership[i] = j;
            }
        }
        min_distance = FLT_MAX;
    }

    return clusters;
}
