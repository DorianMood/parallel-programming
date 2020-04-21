#!/bin/bash

echo -ne "BUILDING..."
cd ./kmeans
output=$(make 2>&1)
if [[ $output != *"make:"* ]]; then
    echo -e "\rBUILDING SUCCESSFUL."
else
    echo -e "\rBUILD ERROR :"
    echo $output
fi

# Run tests for K-Means
for nobjects in 10000 100000 1000000; do
    for nclusters in 2 4 8 16; do
        # Sequential
        ./kmeans-serial -s $nobjects -k $nclusters
        # Parallels
        for nthreads in 1 2 4 8 16 32 64; do
            ./kmeans-parallel -s $nobjects -k $nclusters -n $nthreads
        done
    done
done


