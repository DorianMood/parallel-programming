#include "Sorters.hh"

#include <vector>
#include <algorithm>
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
    uint64_t output[array_size];
    int i;
    uint64_t count[10] = {0};

    printf("%jd", exp);

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
        countSort(array, array_size, exp);
}

void ParallelRadixSorter::sort(uint64_t *array, int array_size)
{
    uint64_t m = getMax(array, array_size);
    
    for (uint64_t exp = 1; m / exp > 0; exp *= 10)
        countSort(array, array_size, exp);
}
