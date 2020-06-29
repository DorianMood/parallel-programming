#include <device_launch_parameters.h>
#include <cstdio>

extern "C" {
    // PUT YOUR KERNEL FUNCTION HERE
    __global__ void bfs_visit_next(
        int* adjacencyList,
        int* edgesOffset,
        int* edgesSize,
        int* distance,
        int* parent,
        int* currentQueue,
        int* nextQueue,
        int queueSize,
        int* nextQueueSize,
        int* degrees,
        int* incrDegrees
    )
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid < queueSize)
        {
            int v = currentQueue[tid];
            for (int i = edgesOffset[v]; i < edgesOffset[v] + edgesSize[v]; i++)
            {
                int u = adjacencyList[i];
                if (parent[u] == -1) // Not visited
                { // Visit
                    parent[u] = v;
                    distance[u] = distance[v] + 1;
    
                    int pos = atomicAdd(nextQueueSize, 1);
                    nextQueue[pos] = u;
                }
            }
        }
    }
}
