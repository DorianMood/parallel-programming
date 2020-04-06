#include "Sorters.hh"

#include <vector>
#include <algorithm>
#include <iostream>

#include <cstdio>

void ShellSorter::sort(uint64_t *array, int array_size)
{
    int gap = array_size / 2;
    for (int gap = array_size / 2; gap > 0; gap /= 2)
    {
        for (int i = gap; i < array_size; i++)
        {
            uint64_t temp = array[i];
            int j;
            for (j = i; j >= gap && array[j - gap] > temp; j -= gap)
                array[j] = array[j - gap];
            array[j] = temp;
        }
    }
}

void ParallelShellSorter::sort(uint64_t *array, int array_size)
{
    // Create thread array
    pthread_t *threads;

    int gap = array_size / 2;
    for (int gap = array_size / 2; gap > 0; gap /= 2)
    {
        int thread_counter = 0;
        for (int i = gap; i < array_size; i++)
        {
            // Clean up threads
            for (int tid = 0; tid < this->m_nthreads; tid++)
                pthread_join(threads[tid], NULL);
            //free(threads);

            threads = (pthread_t *)malloc(this->m_nthreads * sizeof(pthread_t));
            for (int tid = 0; tid < this->m_nthreads; tid++)
                pthread_create(
                    &threads[tid],
                    NULL,
                    &(this->thread_create_helper),
                    new ParallelShellSorterArgs(this, tid));
        }
        // Clean up threads
        for (int tid = 0; tid < this->m_nthreads; tid++)
            pthread_join(threads[tid], NULL);
    }
    // Clean up threads
    for (int tid = 0; tid < this->m_nthreads; tid++)
        pthread_join(threads[tid], NULL);
    //free(threads);
}

void *ParallelShellSorter::thread_body(void *arg)
{
    int tid = ((ParallelShellSorterArgs *)arg)->tid;
    printf("%d", tid);

    // uint64_t temp = array[i];
    // int j;
    // for (j = i; j >= gap && array[j - gap] > temp; j -= gap)
    //     array[j] = array[j - gap];
    // array[j] = temp;
    return NULL;
}
