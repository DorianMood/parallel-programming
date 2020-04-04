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
