#include <cstdio>

#include <device_launch_parameters.h>
#include <cuda.h>

#include <map>
#include <vector>
#include <math.h>

#include "base.h"

#define BLOCK_DIM 256

extern "C"
{
    /**
     * Compute distances from each B point to each A point
     * @param A pointer to 1D feature refence vector
     * @param size_A size of vector A
     * @param B pointer to 1D feature query vector of dimensions
     * @param size_B size of vector B
     * @param D output vector of size_B * DIMENSION
     * */
    __global__ void cuda_compute_distance(float *A, int size_A,
                                          float *B, int size_B,
                                          float *D)
    {
        // Gloabal thread ID
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        int query_id = tid;

        if (tid < size_B)
        {
            // Get distances
            for (int i = 0; i < size_A; i++)
            {
                float sum = 0.0;
                for (int j = 0; j < DIMENSION; j++)
                {
                    sum += pow(A[DIMENSION * i + j] - B[tid * DIMENSION + j], 2);
                }
                D[tid * size_A + i] = sqrt(sum);
            }
        }
        __syncthreads();
    }

    __global__ void cuda_get_classes(float *distances, int num_r, int num_q,
                                int *classes
    )
    {
        // Gloabal thread ID
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int query_id = tid;

        if (query_id < num_q)
        {
            // Get class for query_id point
            int point_class = 0;

            // Iterate over all the distances from query_id point
            for (int i = 0; i < num_r; i++)
            {
                distances[query_id * num_r + i];
            }

            // Set corresponding class
            classes[num_r + query_id] = 0;
        }
        __syncthreads();
    }
}
