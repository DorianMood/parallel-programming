#include "Sorters.hh"

#include <vector>
#include <algorithm>

void RadixSorter::sort(uint64_t* array, int array_size) 
{
  std::vector<uint64_t> vec;
  for( int i=0;i<array_size;i++) {
    vec.push_back(array[i]);
  }
  std::sort(vec.begin(), vec.end());
  for( int i=0;i<array_size;i++) {
    array[i] = vec[i];
  }
}

void ParallelRadixSorter::sort(uint64_t* array, int array_size) 
{
  std::vector<uint64_t> vec;
  for( int i=0;i<array_size;i++) {
    vec.push_back(array[i]);
  }
  std::sort(vec.begin(), vec.end());
  for( int i=0;i<array_size;i++) {
    array[i] = vec[i];
  }
}
