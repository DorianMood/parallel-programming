#ifndef _H_KMEANS
#define _H_KMEANS

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

/* cluster.c */
int     cluster(int, int, float**, int, float, float***);

/* kmeans_clustering.c */
float **serial_clustering(float**, int, int, int, float, int*);
float **parallel_clustering(float**, int, int, int, float, int*);

float   euclid_dist_2        (float*, float*, int);
int     find_nearest_point   (float* , int, float**, int);

#endif
