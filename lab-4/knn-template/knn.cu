#include <cstdio>

#include <device_launch_parameters.h>
#include <cuda.h>

#include <map>
#include <vector>
#include <math.h>

#include "base.h"

#define BLOCK_DIM 256

// extern "C"
// {
    /**
     * Compute distances from each B point to each A point
     * @param ref pointer to 1D feature refence vector
     * @param size_r size of vector reference
     * @param query pointer to 1D feature query vector of dimensions
     * @param size_q size of vector query
     * @param distance output vector of size_r * size_q
     * */
    __global__ void cuda_compute_distance(float *coords, int size_r,
                                          float *newCoords, int size_q,
                                          float *distance)
    {
        // Gloabal thread ID
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        int query_id = tid;

        if (tid < size_q)
        {
            // Get distances
            for (int i = 0; i < size_r; i++)
            {
                float sum = 0.0;
                for (int j = 0; j < DIMENSION; j++)
                {
                    sum += (coords[DIMENSION * i + j] - newCoords[tid * DIMENSION + j]) * (coords[DIMENSION * i + j] - newCoords[tid * DIMENSION + j]);
                }
                distance[tid * size_r + i] = sqrt(sum);
            }
        }
    }

    /**
    * For each reference point (i.e. each column) finds the k-th smallest distances
    * of the distance matrix and their respective indexes and gathers them at the top
    * of the 2 arrays.
    *
    * Since we only need to locate the k smallest distances, sorting the entire array
    * would not be very efficient if k is relatively small. Instead, we perform a
    * simple insertion sort by eventually inserting a given distance in the first
    * k values.
    *
    * @param dist         distance matrix
    * @param dist_pitch   pitch of the distance matrix given in number of columns
    * @param index        index matrix
    * @param index_pitch  pitch of the index matrix given in number of columns
    * @param width        width of the distance matrix and of the index matrix
    * @param height       height of the distance matrix
    * @param k            number of values to find
    */
    __global__ void modified_insertion_sort(float * dist,
                                            int *   classes,
                                            int     size_r,
                                            int     size_q,
                                            int     k,
                                            int     num_classes
    )
    {
        // Column position
        unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

        // For this query point
        if (xIndex < size_q)
        {
            // Pointer shift
            float * p_dist  = dist  + xIndex * size_r;

            // Initialise the top classes array
            int *top_classes = new int[k];
            for (int i = 0; i < k; i++)
                top_classes[i] = classes[size_r + i];

            // Iterate through all points
            for (int i = 1; i < size_r; ++i)
            {
                // Store current distance and associated index
                float curr_dist = p_dist[i];
                int curr_class = classes[i];

                // Skip the current value if its index is >= k and if it's higher the k-th already sorted mallest value
                if (curr_dist >= p_dist[(k - 1)])
                    continue;

                // Shift values (and indexes) higher that the current distance to the right
                int j = min(i, k - 1);
                while (j > 0 && p_dist[j - 1] > curr_dist)
                {
                    p_dist[j] = p_dist[j - 1];
                    top_classes[j] = top_classes[j - 1];
                    --j;
                }

                // Write the current distance and index at their position
                p_dist[j]   = curr_dist;
                top_classes[j] = curr_class; 
            }

            // Get class for current point accourding to the top classes
            // Implement majority vote
            classes[size_r + xIndex] = top_classes[0];
            int classes_sum = 0, classes_max = 0;
            for (int i = 0; i < num_classes; i++)
            {
                classes_sum = 0;
                for (int j = 0; j < k; j++)
                    if (i == top_classes[j])
                        classes_sum += 1;
                if (classes_sum > classes_max)
                {
                    classes[size_r + xIndex] = i;
                    classes_max = classes_sum;
                }
            }
        }
    }
//}
