#include <device_launch_parameters.h>
#include <cstdio>


extern "C" {
    // PUT YOUR KERNEL FUNCTION HERE
    __global__ void bfs(Graph &G)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        //if (tid < G.numVertices)
        //{
            // process vertice
        //}
    }
}
