#include <cstdio>
#include <cuda.h>
#include <string>

#include "graph.h"
#include "bfsCPU.h"
#include "bfsCUDA.cu"
#include "hwtimer.h"

#define GPU_DEVICE 0

void runCpu(int startVertex, Graph &G, std::vector<int> &distance,
            std::vector<int> &parent, std::vector<bool> &visited) {
    bfsCPU(startVertex, G, distance, parent, visited);
}

void checkError(cudaError_t  error, std::string msg) {
    if (error != cudaSuccess) {
        printf("%s: %d\n", msg.c_str(), error);
        exit(1);
    }
}

cudaDeviceProp deviceProp;

int* d_adjacencyList;
int* d_edgesOffset;
int* d_edgesSize;
int* d_distance;
int* d_parent;
int* d_currentQueue;
int* d_nextQueue;
int* d_degrees;
int* incrDegrees;

void initCuda(Graph &G) {
    //initialize CUDA
    checkError(cudaGetDeviceProperties(&deviceProp, GPU_DEVICE), "cannot get device");
    printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
    checkError(cudaSetDevice(GPU_DEVICE), "cannot set device");

    //copy memory to device
    checkError(cudaMalloc(&d_adjacencyList, G.numEdges * sizeof(int)), "cannot allocate d_adjacencyList");
    checkError(cudaMalloc(&d_edgesOffset, G.numVertices * sizeof(int)), "cannot allocate d_edgesOffset");
    checkError(cudaMalloc(&d_edgesSize, G.numVertices * sizeof(int)), "cannot allocate d_edgesSize");
    checkError(cudaMalloc(&d_distance, G.numVertices * sizeof(int)), "cannot allocate d_distance");
    checkError(cudaMalloc(&d_parent, G.numVertices * sizeof(int)), "cannot allocate d_parent");
    checkError(cudaMalloc(&d_currentQueue, G.numVertices * sizeof(int)), "cannot allocate d_currentQueue");
    checkError(cudaMalloc(&d_nextQueue, G.numVertices * sizeof(int)), "cannot allocate d_nextQueue");
    checkError(cudaMalloc(&d_degrees, G.numVertices * sizeof(int)), "cannot allocate d_degrees");
    checkError(cudaMallocHost((void **) &incrDegrees, sizeof(int) * G.numVertices), "cannot allocate memory");

    checkError(cudaMemcpy(d_adjacencyList, G.adjacencyList.data(), G.numEdges * sizeof(int), cudaMemcpyHostToDevice),
               "cannot copy to d_adjacencyList");
    checkError(cudaMemcpy(d_edgesOffset, G.edgesOffset.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice),
               "cannot copy to d_edgesOffset");
    checkError(cudaMemcpy(d_edgesSize, G.edgesSize.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice),
               "cannot copy to d_edgesSize");
}

void finalizeCuda() {
    //free memory
    checkError(cudaFree(d_adjacencyList), "cannot free memory for d_adjacencyList");
    checkError(cudaFree(d_edgesOffset), "cannot free memory for d_edgesOffset");
    checkError(cudaFree(d_edgesSize), "cannot free memory for d_edgesSize");
    checkError(cudaFree(d_distance), "cannot free memory for d_distance");
    checkError(cudaFree(d_parent), "cannot free memory for d_parent");
    checkError(cudaFree(d_currentQueue), "cannot free memory for d_parent");
    checkError(cudaFree(d_nextQueue), "cannot free memory for d_parent");
    checkError(cudaFreeHost(incrDegrees), "cannot free memory for incrDegrees");
}

void checkOutput(std::vector<int> &distance, std::vector<int> &expectedDistance, Graph &G) {
    for (int i = 0; i < G.numVertices; i++) {
        if (distance[i] != expectedDistance[i]) {
            printf("%d %d %d\n", i, distance[i], expectedDistance[i]);
            printf("Wrong output!\n");
            exit(1);
        }
    }

    printf("Output OK!\n\n");
}

void initializeCudaBfs(int startVertex, std::vector<int> &distance, std::vector<int> &parent, Graph &G) {
    //initialize values
    std::fill(distance.begin(), distance.end(), std::numeric_limits<int>::max());
    std::fill(parent.begin(), parent.end(), std::numeric_limits<int>::max());
    distance[startVertex] = 0;
    parent[startVertex] = 0;

    checkError(cudaMemcpy(d_distance, distance.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice),
               "cannot copy to d)distance");
    checkError(cudaMemcpy(d_parent, parent.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice),
               "cannot copy to d_parent");

    int firstElementQueue = startVertex;
    cudaMemcpy(d_currentQueue, &firstElementQueue, sizeof(int), cudaMemcpyHostToDevice);
}

void finalizeCudaBfs(std::vector<int> &distance, std::vector<int> &parent, Graph &G) {
    //copy memory from device
    checkError(cudaMemcpy(distance.data(), d_distance, G.numVertices * sizeof(int), cudaMemcpyDeviceToHost),
               "cannot copy d_distance to host");
    checkError(cudaMemcpy(parent.data(), d_parent, G.numVertices * sizeof(int), cudaMemcpyDeviceToHost), "cannot copy d_parent to host");

}

void runCudaBfs(int startVertex, Graph &G, std::vector<int> &distance,
                      std::vector<int> &parent) {
    initializeCudaBfs(startVertex, distance, parent, G);

    // PUT YOUR PARALLEL CODE HERE
    /*
       1. Design the BFS parallel code.
       1. Specify the sizes of grid and block.
       2. Launch the kernel function (Write kernel code in bfsCUDA.cu).
    */

    // put distance and parent to shared memory

    bfs<<<1, G.adjacencyList.size()>>>(G);

    // load distance and parent back
    
    finalizeCudaBfs(distance, parent, G);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("usage: ./bfs-exec <start vertex> <number of vertices> <number of edges>\n");
        exit(1);
    }

    hwtimer_t timer;
    initTimer(&timer);
    
    // read graph from standard input
    Graph G;
    int startVertex = atoi(argv[1]);

    readGraph(G, argc, argv);

    printf("Number of vertices %d\n", G.numVertices);
    printf("Number of edges %d\n\n", G.numEdges);

    //vectors for results
    std::vector<int> distance(G.numVertices, std::numeric_limits<int>::max());
    std::vector<int> parent(G.numVertices, std::numeric_limits<int>::max());
    std::vector<bool> visited(G.numVertices, false);
 
    printf("Starting sequential bfs.\n");
    startTimer(&timer);
    //run CPU sequential bfs
    runCpu(startVertex, G, distance, parent, visited);
    stopTimer(&timer);
    printf("Elapsed time: %lld ns.\n\n", getTimerNs(&timer));

    //save results from sequential bfs
    std::vector<int> expectedDistance(distance);
    std::vector<int> expectedParent(parent);

    initCuda(G);
    //run CUDA simple parallel bfs
    printf("Starting parallel bfs.\n");
    startTimer(&timer);
    runCudaBfs(startVertex, G, distance, parent);
    stopTimer(&timer);
    printf("Elapsed time: %lld ns.\n\n", getTimerNs(&timer));

    checkOutput(distance, expectedDistance, G);

    finalizeCuda();
    return 0;
}
