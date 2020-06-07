#include <cstdio>

#include <device_launch_parameters.h>
#include <cuda.h>
#include <thrust/sort.h>

#include <map>
#include <vector>
#include <math.h>

#include "base.h"

#define BLOCK_DIM 256

extern "C"
{

    __global__ void cuInsertionSort(float *dist, long *ind, int width, int height, int k)
    {

        // Variables
        int l, i, j;
        float *p_dist;
        long *p_ind;
        float curr_dist, max_dist;
        long curr_row, max_row;
        unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

        if (xIndex < width)
        {
            // Pointer shift, initialization, and max value
            p_dist = dist + xIndex;
            p_ind = ind + xIndex;
            max_dist = p_dist[0];
            p_ind[0] = 1;

            // Part 1 : sort kth firt elementZ
            for (l = 1; l < k; l++)
            {
                curr_row = l * width;
                curr_dist = p_dist[curr_row];
                if (curr_dist < max_dist)
                {
                    i = l - 1;
                    for (int a = 0; a < l - 1; a++)
                    {
                        if (p_dist[a * width] > curr_dist)
                        {
                            i = a;
                            break;
                        }
                    }
                    for (j = l; j > i; j--)
                    {
                        p_dist[j * width] = p_dist[(j - 1) * width];
                        p_ind[j * width] = p_ind[(j - 1) * width];
                    }
                    p_dist[i * width] = curr_dist;
                    p_ind[i * width] = l + 1;
                }
                else
                {
                    p_ind[l * width] = l + 1;
                }
                max_dist = p_dist[curr_row];
            }

            // Part 2 : insert element in the k-th first lines
            max_row = (k - 1) * width;
            for (l = k; l < height; l++)
            {
                curr_dist = p_dist[l * width];
                if (curr_dist < max_dist)
                {
                    i = k - 1;
                    for (int a = 0; a < k - 1; a++)
                    {
                        if (p_dist[a * width] > curr_dist)
                        {
                            i = a;
                            break;
                        }
                    }
                    for (j = k - 1; j > i; j--)
                    {
                        p_dist[j * width] = p_dist[(j - 1) * width];
                        p_ind[j * width] = p_ind[(j - 1) * width];
                    }
                    p_dist[i * width] = curr_dist;
                    p_ind[i * width] = l + 1;
                    max_dist = p_dist[max_row];
                }
            }
        }
    }

    __device__ void cuParallelSqrt(float *dist, int width, int k)
    {
        unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
        if (xIndex < width && yIndex < k)
            dist[yIndex * width + xIndex] = sqrt(dist[yIndex * width + xIndex]);
    }

    __device__ void cuda_compute_distances(float *A, int size_A,
                                           float *B, int size_B,
                                           float *D, int dim)
    {
        
    }

    /**
     * Function to calculate KNN on GPU.
     * 
    @param samples pointer to 1D sample features array
    @param numSamples number of sample features
    @param newSamples pointer to 1D query features array
    @param numNewSamples number of query features
    @param classes pointer to 1D classes array
    @param numClasses number of clusters
    @param k k-NN k parameter, number of newrest neighbours

    **/
    __global__ void knn(
        float *samples,
        int numSamples,
        float *newSamples,
        int numNewSamples,
        int* classes,
        int numClasses,
        int k
    )
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        double *distances = new double[numNewSamples];

        // Classify every single new sample per thread
        if (tid < numNewSamples)
        {
            // Calculate distances

            // Sort classes by distances

            // Calculate classes frequencies

            // Get class with the greatest frequency

            // Write class to output
        }
    }
}
