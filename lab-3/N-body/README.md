## General

Sequential and Parallel N-body using MPI


## Compilation

`make clean` removes the compiled executable `nbody`.

`make` compiles the code.

## Running

`mpirun -np <nProcess> ./nbody <nParticle> <nTimestep> <sizeTimestep>`.

`<nProcess>` is the desired number of processes,

`<nParticle>` is the number of particles,

`<nTimestep>` is the number of time steps.

`<sizeTimestep>` is the length of the time step.

Note that `<nParticle>` % `<nProcess>` == 0.
