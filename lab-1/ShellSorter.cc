#include "Sorters.hh"

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

    for (int gap = array_size / 2; gap > 0; gap /= 2)
    {
        this->gap = gap;
        // Create threads
        for (int tid = 0; tid < m_nthreads; tid++)
        {
            int thread_error = pthread_create(
                &threads[tid],
                NULL,
                &(thread_create_helper),
                new ParallelShellSorterArgs(this, tid));
        }
        // Clean up threads
        for (int tid = 0; tid < m_nthreads; tid++)
            pthread_join(threads[tid], NULL);
    }
    free(threads);
}

void *ParallelShellSorter::thread_body(void *arg)
{
    ParallelShellSorterArgs *args = (ParallelShellSorterArgs *)arg;

    int tid = args->tid;
    int gap, j, step;

    gap = this->gap;

    if (tid >= gap)
        return NULL;

    step = gap > m_nthreads ? m_nthreads : gap;

    for (int i = gap + tid; i < array_size; i += step)
    {
        uint64_t temp = (*array)[i];
        for (j = i; j >= gap && (*array)[j - gap] > temp; j -= gap)
            (*array)[j] = (*array)[j - gap];
        (*array)[j] = temp;
    }
    
    return NULL;
}
