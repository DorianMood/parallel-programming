#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <iostream>

#include <thrust/sort.h>

#include "knn.cu"

#define DEBUG 1

void check_error(cudaError_t err, const char *msg);
void knnParallel(float *coords, float *newCoords, int *classes, int numClasses, int numSamples, int numNewSamples, int k);

void knnParallel(float *coords, float *newCoords, int *classes, int numClasses, int numSamples, int numNewSamples, int k)
{
    //*** Device-variables-declaration ***
    float *d_coords;
    float *d_newCoords;
    int *d_classes;
    float *d_distances;

    int totalSamples = numSamples + numNewSamples;

    //*** device-allocation ***
    check_error(cudaMalloc(&d_coords, totalSamples * DIMENSION * sizeof(float)), "alloc d_coords_x");
    check_error(cudaMalloc(&d_classes, totalSamples * sizeof(int)), "alloc d_classes");
    check_error(cudaMalloc(&d_newCoords, numNewSamples * DIMENSION * sizeof(float)), "alloc d_coordsnew");
    check_error(cudaMalloc(&d_distances, numNewSamples * numSamples * sizeof(float)), "alloc d_distances");

    //***copy-arrays-on-device***
    check_error(cudaMemcpy(d_coords, coords, totalSamples * DIMENSION * sizeof(float), cudaMemcpyHostToDevice), "copy d_coords");
    check_error(cudaMemcpy(d_classes, classes, totalSamples * sizeof(int), cudaMemcpyHostToDevice), "copy d_classes");
    check_error(cudaMemcpy(d_newCoords, newCoords, numNewSamples * DIMENSION * sizeof(float), cudaMemcpyHostToDevice), "copy d_coordsnew");

    const int PROBLEM_SIZE = numNewSamples;
    const int NUM_THREADS = 256;
    const int NUM_BLOCKS = (int)ceil(PROBLEM_SIZE / NUM_THREADS);

    // Calculate distances
    // For performance reasons we can compute just x^2 + y^2 (anyway we are looking for minimum)
    cuda_compute_distance<<<NUM_BLOCKS, NUM_THREADS>>>(d_coords, numSamples, d_newCoords, numNewSamples, d_distances);
    // Sort classes by distances
    //thrust::sort_by_key(d_distances, d_distances + 10, d_classes);

    // Calculate classes frequencies

    // Get class with the highest frequency

    // Write class to output



    cudaDeviceSynchronize();
    // download device -> host
    check_error(cudaMemcpy(coords, d_coords, DIMENSION * totalSamples * sizeof(float), cudaMemcpyDeviceToHost), "download coords");
    check_error(cudaMemcpy(classes, d_classes, totalSamples * sizeof(int), cudaMemcpyDeviceToHost), "download classes");


}

void check_error(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s : error %d (%s)\n", msg, err, cudaGetErrorString(err));
        exit(err);
    }
}
