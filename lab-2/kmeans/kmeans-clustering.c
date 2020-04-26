#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <stdbool.h>

#include "kmeans.h"

extern double wtime(void);
extern int num_omp_threads;

// Calculates distance between two points
float euclid_dist_2(const float *a, const float *b, const int ndimention)
{
    float summa = 0.0;
    for (int i = 0; i < ndimention; i++)
    {
        summa += pow(a[i] - b[i], 2);
    }
    return sqrt(summa);
}

// Returns index of the nearest point for fixed point
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

// Returns arythmetical average for array[N]
float *average(float **elements, int nelements, int ndimension)
{
    float *point = (float *)malloc(ndimension * sizeof(float));
    for (int i = 0; i < nelements; i++)
        for (int j = 0; j < ndimension; j++)
            point[j] += elements[i][j] / (float)nelements;
    return point;
}

// Returns new centroids for given data
void calculate_centroids(float **clusters, // Clusters centroids
                         const int nclusters, // N classes
                         float **feature, // Data
                         const int npoints, // N data points
                         const int nfeatures, // N dimension
                         const int *membership) // Membership of each instance
{
    if (membership[0] == -1)
    {
        int n = 0;
        for (int i = 0; i < nclusters; i++)
        {
            // n = (int)rand() % npoints;
            for (int j = 0; j < nfeatures; j++)
                clusters[i][j] = (float)feature[n][j];
            n++;
        }
        return;
    }

    // Reset cluster centers
    for (int i = 0; i < nclusters; i++)
        for (int j = 0; j < nfeatures; j++)
            clusters[i][j] = 0;

    // Calculate clusters sizes
    int *clusters_size = (int *)malloc(nclusters * sizeof(int));
    for (int i = 0; i < nclusters; i++)
        clusters_size[i] = 0;
    for (int i = 0; i < npoints; i++)
        clusters_size[membership[i]]++;

    // Calculate new cluster centers
    for (int i = 0; i < npoints; i++)
        for (int j = 0; j < nfeatures; j++)
            clusters[membership[i]][j] +=
                feature[i][j]/ (float)clusters_size[membership[i]];

    free(clusters_size);
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
    //clusters[0] = (float *)malloc(nfeatures * sizeof(float));
    for (int i = 0; i < nclusters; i++)
        clusters[i] = (float *)malloc(nfeatures * sizeof(float));

    /* Initilize membership */
    for (int i = 0; i < npoints; i++)
        membership[i] = -1;

    // PUT YOUR CODE HERE

    
    int count;
    do
    {
        count = 0;
        // Calculate new centers
        calculate_centroids(clusters, nclusters, feature, npoints, nfeatures, membership);
        // Perform clustering
        for (int i = 0; i < npoints; i++)
        {
            int cluster_index = (membership[i] == -1) ? 0 : membership[i];
            float min_distance = euclid_dist_2(feature[i], clusters[cluster_index], nfeatures);
            int changed = 0;
            // each cluster
            for (int j = 0; j < nclusters; j++)
            {
                // Look for minimal distance d (feature[i], clusters[j])
                float distance = euclid_dist_2(feature[i], clusters[j], nfeatures);
                if (membership[i] == -1 || distance < min_distance)
                {
                    min_distance = distance;
                    if (membership[i] != j)
                        changed++;
                    membership[i] = j;
                }
            }
            if (changed)
                count++;
        }
    } while ((count / (float)npoints) > threshold);

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
    for (int i = 0; i < nclusters; i++)
        clusters[i] = (float *)malloc(nfeatures * sizeof(float));

    /* Initilize membership */
    for (int i = 0; i < npoints; i++)
        membership[i] = -1;

    // PUT YOUR CODE HERE
    
    int count;
    do
    {
        count = 0;
        // Calculate new centers
        calculate_centroids(clusters, nclusters, feature, npoints, nfeatures, membership);
        // Perform clustering
        // each point
        for (int i = 0; i < npoints; i++)
        {
            int cluster_index = (membership[i] == -1) ? 0 : membership[i];
            float min_distance = euclid_dist_2(feature[i], clusters[cluster_index], nfeatures);
            int changed = 0;
            // each cluster
            for (int j = 0; j < nclusters; j++)
            {
                // Look for minimal distance d (feature[i], clusters[j])
                float distance = euclid_dist_2(feature[i], clusters[j], nfeatures);
                if (membership[i] == -1 || distance < min_distance)
                {
                    min_distance = distance;
                    if (membership[i] != j)
                        changed++;
                    membership[i] = j;
                }
            }
            if (changed)
                count++;
        }
    } while ((count / (float)npoints) > threshold);

    return clusters;
}