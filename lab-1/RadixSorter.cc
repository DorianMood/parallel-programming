#include "Sorters.hh"

#include <cstdio>

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
    int i;
    uint64_t output[array_size];
    uint64_t count[10] = {0};

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

void RadixSorter::sort(uint64_t *array, int array_size)
{
    uint64_t m = getMax(array, array_size);
    for (uint64_t exp = 1; m / exp > 0; exp *= 10)
    {
        int i;
        uint64_t output[array_size];
        uint64_t count[10] = {0};

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
    this->exp = 1;

    // Create threads
    for (int tid = 0; tid < array_size && tid < m_nthreads; tid++)
    {
        printf("%d\t", tid);
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
    ParallelRadixSorterArgs *args = (ParallelRadixSorterArgs *)arg;
    int tid = args->tid;

    if (tid >= array_size)
        return NULL;
    
    uint64_t m = getMax(*array, array_size);

    for (uint64_t exp = 1; m / exp > 0; exp *= 10)
    {
        pthread_barrier_wait(&barrier[0]);
        if (!pthread_mutex_trylock(&mutex[0]))
            for (int i = 0; i < 10; i++)
                count[i] = 0;
        pthread_barrier_wait(&barrier[1]);
        pthread_mutex_unlock(&mutex[0]);
        // Sync here

        // Async code
        for (int i = tid; i < array_size; i += m_nthreads)
        {
            pthread_mutex_lock(mutex);
            count[((*array)[i] / exp) % 10]++;
            pthread_mutex_unlock(mutex);
        }

        pthread_barrier_wait(&barrier[0]);
        // Sync code
        if (!pthread_mutex_trylock(&mutex[1]))
            for (int i = 1; i < 10; i++)
                count[i] += count[i - 1];
        pthread_barrier_wait(&barrier[2]);
        pthread_mutex_unlock(&mutex[1]);

        // Async code
        for (int i = array_size - 1 - tid; i >= 0; i -= m_nthreads)
        {
            pthread_mutex_lock(mutex);
            output[count[((*array)[i] / exp) % 10] - 1] = (*array)[i];
            count[((*array)[i] / exp) % 10]--;
            pthread_mutex_unlock(mutex);
        }

        pthread_barrier_wait(&barrier[3]);

        // Async code
        for (int i = tid; i < array_size; i += m_nthreads)
            (*array)[i] = output[i];
        pthread_barrier_wait(&barrier[3]);
    }
    return NULL;
}
