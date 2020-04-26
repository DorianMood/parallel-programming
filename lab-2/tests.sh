#!/bin/bash

echo -ne "BUILDING K-MEANS..."
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
        echo "sequential";
        ./kmeans-serial -s $nobjects -k $nclusters
        echo "parallel";
        # Parallels
        for nthreads in 1 2 4 8 16 32 64; do
            ./kmeans-parallel -s $nobjects -k $nclusters -n $nthreads
        done
    done
done

cd ..

echo -ne "BUILDING OCEAN..."
cd ./ocean-simulation
output=$(make 2>&1)
if [[ $output != *"make:"* ]]; then
    echo -e "\rBUILDING SUCCESSFUL."
else
    echo -e "\rBUILD ERROR :"
    echo $output
fi
# Run tests for OCEAN
for dim in 18 66 258 1026 4098; do
    for timesteps in 4 16 64 256; do
        echo -n "$dim $timesteps "
        ./serial_ocean -d $dim -t $timesteps
        for nthreads in 1 2 4 8 16 32 64; do
            echo -n "$dim $timesteps $nthreads "
            ./omp_ocean -d $dim -t $timesteps -n $nthreads
        done
    done
done;

