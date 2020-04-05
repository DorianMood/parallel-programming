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

    // Create thread handlers array
    pthread_t *thread_handlers;

    int gap = array_size / 2;
    for (int gap = array_size / 2; gap > 0; gap /= 2)
    {
        int thread_counter = 0;
        for (int i = gap; i < array_size; i++)
        {
            // Clean up threads
            if (thread_counter >= this->m_nthreads)
            {
                std::cout << "FREE" << std::endl;
                for (int t_id = 0; t_id < this->m_nthreads; t_id++)
                    pthread_join(thread_handlers[t_id], NULL);
                thread_counter = 0;
                free(thread_handlers);
            }

            thread_handlers = (pthread_t *)malloc(this->m_nthreads * sizeof(pthread_t));
            for (int t_id = 0; t_id < this->m_nthreads; t_id++)
                pthread_create(
                    &thread_handlers[t_id],
                    NULL,
                    &(this->thread_create_helper),
                    new ParallelShellSorterArgs(this, t_id));
            // uint64_t temp = array[i];
            // int j;
            // for (j = i; j >= gap && array[j - gap] > temp; j -= gap)
            //     array[j] = array[j - gap];
            // array[j] = temp;
        }
    }
}

void *ParallelShellSorter::thread_body(void *arg)
{
    std::cout << "Thread ID : " << ((ParallelShellSorterArgs *)arg)->tid << std::endl;
    return NULL;
}
