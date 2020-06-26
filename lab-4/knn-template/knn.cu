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
     * @param A pointer to 1D feature vector
     * @param size_A size of vector A
     * @param B pointer to 1D feature vector of dimensions
     * @param size_B size of vector B
     * @param D output vector of size_B * DIMENSION
     * */
    __global__ void cuda_compute_distance(float *A, int size_A,
                                          float *B, int size_B,
                                          float *D)
    {
        // Gloabal thread ID
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid < size_B)
        {
            for (int i = 0; i < size_A; i++)
            {
                float sum = 0.0;
                for (int j = 0; j < DIMENSION; j++)
                {
                    sum += pow(A[DIMENSION * i + j] - B[tid * DIMENSION + j], 2);
                }
                D[tid * size_B + i] = sqrt(sum);
            }
        }
        __syncthreads();
    }
    /**
    * 
    * 
    */
    __global__ void cuda_get_class()
    {

    }
}
