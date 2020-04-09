#include "Sorters.hh"

#include <vector>
#include <algorithm>

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
    initialize_shared_memory(array, array_size);

    // Create thread array
    pthread_t *threads;
    threads = (pthread_t *)malloc(m_nthreads * sizeof(pthread_t));

    // for (int tid = 0; tid < m_nthreads; tid++)
    //     pthread_create(
    //         &threads[tid],
    //         NULL,
    //         &thread_create_helper,
    //         new ParallelShellSorterArgs(this, tid)
    //     );
    
    // for (int tid = 0; tid < m_nthreads; tid++)
    //     pthread_join(threads[tid], NULL);
    

    for (int gap = array_size / 2; gap > 0; gap /= 2)
    {
        this->gap = gap;
        gap_offset_i = 0;
        printf("Gap : %d\n", gap);
        // Create threads
        for (int tid = 0; tid < m_nthreads; tid++)
            pthread_create(
                &threads[tid],
                NULL,
                &(thread_create_helper),
                new ParallelShellSorterArgs(this, tid));
        // Clean up threads
        for (int tid = 0; tid < m_nthreads; tid++)
            pthread_join(threads[tid], NULL);
    }
    free(threads);
}

void *ParallelShellSorter::thread_body(void *arg)
{
    int tid = ((ParallelShellSorterArgs *)arg)->tid;
    int gap, i, j;

    gap = this->gap;

    while (true)
    {
        pthread_mutex_lock(&gap_cycle_lock);
        
        printf("%d thread.", tid);
        if (gap_offset_i < (array_size - 1))
            gap_offset_i++;
        else
        {
        printf("%d thread inside else.", tid);
            pthread_mutex_unlock(&gap_cycle_lock);
            break;
        }
        printf("%d :: Gap offset : %d\n", tid, gap_offset_i);
        i = gap_offset_i;
        pthread_mutex_unlock(&gap_cycle_lock);

        uint64_t temp = (*array)[i];
        for (j = i; j >= gap && (*array)[j - gap] > temp; j -= gap)
            (*array)[j] = (*array)[j - gap];
        (*array)[j] = temp;
    }
    return NULL;
}
