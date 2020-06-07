#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "base.h"
#include "knnCPU.h"
#include "knnCUDA.cu"
#include "hwtimer.h"

void knnInit(float* coords, float* newCoords, int* classes, int numSamples, int numClasses, int numNewSamples);
void genRandCoords(float* x, int numSamples);
void checkOutput(float* classes, float* gpuClasses, int numClasses, int totalSamples);

void checkOutput(int* classes, int* gpuClasses, int numClasses, int totalSamples) {
    int* numCpuClasses = (int*)malloc(sizeof(int) * numClasses);
    int* numGpuClasses = (int*)malloc(sizeof(int) * numClasses);

    for (int j = 0; j < numClasses; j++) {
        numCpuClasses[j] = 0;
        numGpuClasses[j] = 0;
    }

    for (int i = 0 ; i < totalSamples; i++) {
        for (int j = 0; j < numClasses; j++) {
            if (classes[i] == j)
                numCpuClasses[j] += 1;
            if (gpuClasses[i] == j)
                numGpuClasses[j] += 1;

        }
    }

    for (int j = 0; j < numClasses; j++) {
        if (numCpuClasses[j] != numGpuClasses[j]) {
            printf("Wrong output!\n");
            exit(1);
        }
    }

    printf("Output OK!\n\n");
}

void knnInit(float* coords, float* newCoords, int* classes, int numSamples, int numClasses, int numNewSamples) {
    for (int i = 0; i < numSamples; i++) 
        classes[i] = rand() % numClasses;

    genRandCoords(coords, numSamples);
    genRandCoords(newCoords, numNewSamples);
}

void genRandCoords(float* x, int numSamples) {
    for (int i = 0; i < numSamples; i++)
        for (int j = 0; j < DIMENSION; j++)
            x[i * DIMENSION + j] = (float)rand()/(float)(RAND_MAX/POINTS_MAX) + (float)(POINTS_MIN);
}

int main(int argc, char** argv) {
    if (argc != 5) {
        printf("usage: ./knn_exec <k nearest neighbors> <number of classes> <number of existing samples> <number of new samples>\n");
        exit(1);
    }

    hwtimer_t timer;
    initTimer(&timer);
    
    int k = atoi(argv[1]); // number of k nearest neighbors
    int numClasses = atoi(argv[2]); // number of classes
    int numSamples = atoi(argv[3]); // number of existing samples
    int numNewSamples = atoi(argv[4]); // number of samples to classify
    int numTotalSamples = numSamples + numNewSamples; // total samples
    
    // array with a class for each sample
    int* classes = (int*)malloc(sizeof(int) * numTotalSamples); 
    float* newCoords = (float*)malloc(sizeof(float) * numNewSamples * DIMENSION);
    float* coords = (float*)malloc(sizeof(float) * numTotalSamples * DIMENSION);

    // gpu samples (initialized from cpu samples)
    int* gpuClasses = (int*)malloc(sizeof(int) * numTotalSamples); 
    float* gpuNewCoords = (float*)malloc(sizeof(float) * numNewSamples * DIMENSION);
    float* gpuCoords = (float*)malloc(sizeof(float) * numTotalSamples * DIMENSION);

    srand(12345);

    printf("Starting initialization.\n");
    startTimer(&timer);
    knnInit(coords, newCoords, classes, numSamples, numClasses, numNewSamples);
    stopTimer(&timer);
    printf("Elapsed time: %lld ns.\n\n", getTimerNs(&timer));

    memcpy(gpuClasses, classes, sizeof(int) * numTotalSamples);
    memcpy(gpuNewCoords, newCoords, sizeof(float) * numNewSamples * DIMENSION);
    memcpy(gpuCoords, coords, sizeof(float) * numTotalSamples * DIMENSION);

    printf("Starting sequential knn.\n");
    startTimer(&timer);
    knnSerial(coords, newCoords, classes, numClasses, numSamples, numNewSamples, k);
    stopTimer(&timer);
    printf("Elapsed time: %lld ns.\n\n", getTimerNs(&timer));

    printf("Starting parallel knn.\n");
    startTimer(&timer);
    knnParallel(gpuCoords, gpuNewCoords, gpuClasses, numClasses, numSamples, numNewSamples, k);
    stopTimer(&timer);
    printf("Elapsed time: %lld ns.\n\n", getTimerNs(&timer));
    
    checkOutput(classes, gpuClasses, numClasses, numTotalSamples);

    free(classes);
    free(newCoords);
    free(coords);
    free(gpuClasses);
    free(gpuNewCoords);
    free(gpuCoords);

    return 0;
}



