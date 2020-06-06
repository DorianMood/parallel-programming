#include <queue>
#include <iostream>

#include "bfsCPU.h"

void bfsCPU(int start, Graph &G, std::vector<int> &distance,
            std::vector<int> &parent, std::vector<bool> &visited) {

    std::queue<int> q;

    q.push(start);
    visited[start] = true;
    distance[start] = 0;
    
    while (!q.empty())
    {
        int v = q.front();
        q.pop();
        for (int i = 0; i < G.edgesSize[v]; i++)
        {
            int u = G.adjacencyList[G.edgesOffset[v] + i];
            if (!visited[u])
            {
                parent[u] = v;
                distance[u] = distance[v] + 1;

                visited[u] = true;
                q.push(u);
            }
        }
    }
}
