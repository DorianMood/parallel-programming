#include "Sorters.hh"

#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdint>

uint64_t getMax(const uint64_t *array, const int array_size)
{
    uint64_t m = array[0];
    for (int i = 0; i < array_size; i++)
        if (array[i] > m)
            m = array[i];
    return m;
}

void countSort(uint64_t *array, int array_size, uint64_t exp)
{
}

void RadixSorter::sort(uint64_t *array, int array_size)
{
    uint64_t m = getMax(array, array_size);
    //printf("%lu\n", m);
    //uint64_t result = m / 10;
    //printf("%lu\t", result);
    for (uint64_t exp = 1; m / exp > 0; exp *= 10)
    {
        uint64_t output[array_size];
        int i;
        uint64_t count[10] = {0};

        //printf("%jd", exp);

        for (i = 0; i < array_size; i++)
            count[(array[i] / exp) % 10]++;

        for (i = 1; i < 10; i++)
            count[i] += count[i - 1];

        for (i = array_size - 1; i >= 0; i--)
        {
            output[count[(array[i] / exp) % 10] - 1] = array[i];
            count[(array[i] / exp) % 10]--;
        }

        for (i = 0; i < array_size; i++)
            array[i] = output[i];
    }
}

void ParallelRadixSorter::sort(uint64_t *array, int array_size)
{
    initialize_shared_memory(array, array_size);
    // Create thread array
    pthread_t *threads;
    threads = (pthread_t *)malloc(m_nthreads * sizeof(pthread_t));

    uint64_t m = getMax(array, array_size);
    this->exp = 1;

    // Create threads
    for (int tid = 0; tid < m_nthreads; tid++)
    {
        int thread_error = pthread_create(
            &threads[tid],
            NULL,
            &(thread_create_helper),
            new ParallelRadixSorterArgs(this, tid));
        if (thread_error)
            printf("ERROR CREATING THREAD\n");
    }
    // Clean up threads
    for (int tid = 0; tid < m_nthreads; tid++)
        pthread_join(threads[tid], NULL);
}

void *ParallelRadixSorter::thread_body(void *arg)
{
    return NULL;
    // TODO : implement following logic in parallel.
    int tid = ((ParallelRadixSorterArgs *)arg)->tid;
    uint64_t m = getMax(*array, array_size);
    
    for (uint64_t exp = 1; m / exp > 0; exp *= 10)
    {
        uint64_t output[array_size];
        int i;
        uint64_t count[10] = {0};
        // Sync here

        // Async code
        // Shared var i
        for (i = 0; i < array_size; i++)
            count[((*array)[i] / exp) % 10]++;
        // Sync code
        for (i = 1; i < 10; i++)
            count[i] += count[i - 1];
        // Async code
        // Shared var i
        for (i = array_size - 1; i >= 0; i--)
        {
            output[count[((*array)[i] / exp) % 10] - 1] = (*array)[i];
            count[((*array)[i] / exp) % 10]--;
        }
        // Async code
        // Shared var i
        for (i = 0; i < array_size; i++)
            (*array)[i] = output[i];
    }
    return NULL;
}
