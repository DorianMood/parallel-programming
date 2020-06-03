#!/bin/bash
'''
echo -ne "BUILDING CG..."
cd ./CG
make clean
output=$(make 2>&1)
if [[ $output != *"make:"* ]]; then
    echo -e "\rBUILDING SUCCESSFUL."
else
    echo -e "\rBUILD ERROR :"
    echo $output
fi

# Run tests for CG
for n in 1024 2048 4096 8192; do
    for step in 50 100 200 500; do
        # Sequential Parallels
        for nthreads in 1 2 4 8 16; do
            echo -n "$nthreads $n $step"
            mpirun -np $nthreads ./cg $n $step
        done
    done
done

cd ..
'''
echo -ne "BUILDING N-BODY..."
cd ./N-body
make clean
output=$(make 2>&1)
if [[ $output != *"make:"* ]]; then
    echo -e "\rBUILDING SUCCESSFUL."
else
    echo -e "\rBUILD ERROR :"
    echo $output
fi

# Run tests for N-BODY
for nParticle in 128 256 512 1024; do
    for nTimestep in 50 100 200 500; do
        for nthreads in 1 2 4 8 16; do
            echo -n "$nthreads $nParticle $nTimestep"
            mpiexec -n $nthreads ./nbody $nParticle $nTimestep 1
        done
    done
done;
