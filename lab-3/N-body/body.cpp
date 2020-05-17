#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "body.h"

void Usage(char *prog_name) // Input Description
{
    fprintf(stderr, "usage: mpiexec -n <nProcesses> %s\n", prog_name);
    fprintf(stderr, "<nParticle> <nTimestep> <sizeTimestep>\n");
    exit(0);
}

void Get_args(int argc, char *argv[], int *n_p, int *n_steps_p, double *delta_t_p) // get parameter information
{
    if (my_rank == 0)
    {
        if (argc != 4)
            Usage(argv[0]);
        *n_p = strtol(argv[1], NULL, 10);
        *n_steps_p = strtol(argv[2], NULL, 10);
        *delta_t_p = strtod(argv[3], NULL);
        if (*n_p <= 0 || *n_p % comm_sz || *n_steps_p < 0 || *delta_t_p <= 0) // illegal inputs
        {
            printf("illegal input\n");
            if (my_rank == 0)
                Usage(argv[0]);
            MPI_Finalize();
            exit(0);
        }
    }
    MPI_Bcast(n_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_steps_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(delta_t_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// the initial conditions are automatically generated, and all processes call this function because of gather communication
void Gen_init_cond(double masses[], vect_t pos[], vect_t loc_vel[], int n, int loc_n)
{
    const double mass = 10000, gap = 0.01, speed = 0;
    if (my_rank == 0)
    {
        int ny = ceil(sqrt(n));
        double y = 0.0;
        for (int i = 0; i < n; i++)
        {
            masses[i] = mass;
            pos[i][X] = i * gap;
            if ((i + 1) % ny == 0)
                ++y;
            pos[i][Y] = y;
            vel[i][X] = 0.0;
            vel[i][Y] = 0.0;
        }
    }
    // Synchronization mass, position information, distribution vector information
    MPI_Bcast(masses, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pos, n, vect_mpi_t, 0, MPI_COMM_WORLD);
    MPI_Scatter(vel, loc_n, vect_mpi_t, loc_vel, loc_n, vect_mpi_t, 0, MPI_COMM_WORLD);
}

void Output_parallel(double masses[], vect_t pos[], vect_t loc_vel[], int n, int loc_n) // output current states
{
    // gather information from all processes for outputs
    MPI_Gather(loc_vel, loc_n, vect_mpi_t, vel, loc_n, vect_mpi_t, 0, MPI_COMM_WORLD);
    if (my_rank == 0)
    {
        printf("Output Parallel State\n");
        for (int i = 0; i < n; i++)
            printf(" %3d X: %.5f Y: %.5f\n", i, pos[i][X], pos[i][Y]);
        printf("\n");
        fflush(stdout);
    }
}

void Output_serial(double masses[], vect_t pos[], vect_t vel[], int n) // output current states
{
    if (my_rank == 0)
    {
        printf("Output Serial State\n");
        for (int i = 0; i < n; i++)
            printf(" %3d X: %.5f Y: %.5f\n", i, pos[i][X], pos[i][Y]);
        printf("\n");
        fflush(stdout);
    }
}

// nbody parallel implementation
void nbody_parallel(double masses[], vect_t loc_forces[], vect_t pos[], vect_t loc_pos[], vect_t loc_vel[], int n, int loc_n, int n_steps, double delta_t)
{
    /* PUT OR MODIFY YOUR PARALLEL CODE IN THIS FUNCTION*/
    int step;
    for (step = 1; step <= n_steps; step++)
    {
        // Calculate the force, update the states, and then synchronize positions
#ifdef DEBUG
        if (step == n_steps)
            Output_parallel(masses, pos, loc_vel, n, loc_n);
#endif
    }
    /* PUT OR MODIFY YOUR PARALLEL CODE IN THIS FUNCTION*/
}

// nbody serial implementation
void nbody_serial(double masses[], vect_t forces[], vect_t pos[], vect_t vel[], int n, int n_steps, double delta_t)
{
    /* PUT OR MODIFY YOUR SERIAL CODE IN THIS FUNCTION*/
    int step;
    for (step = 1; step <= n_steps; step++)
    {
        // Calculate the force, update the states, and then synchronize positions
#ifdef DEBUG
        if (step == n_steps)
            Output_serial(masses, pos, vel, n);
#endif
    }
    /* PUT OR MODIFY YOUR SERIAL CODE IN THIS FUNCTION*/

    const double G = 9.8;
    vect_t *a = new vect_t[n];

    // calculate new forces
    for (int i = 0; i < n; i++)
    {
        // Current force to zero
        forces[i][X] = 0;
        forces[i][Y] = 0;

        for (int j = 0; j < n; j++)
        {
            // Calculate a single force between i-th and j-th bodies
            vect_t d, f;

            double dividend = G * masses[i] * masses[j];
            
            d[X] = abs(pos[i][X] - pos[j][X]);
            f[X] = dividend / pow(d[X], 2);

            d[Y] = abs(pos[i][Y] - pos[j][Y]);
            f[Y] = dividend / pow(d[Y], 2);

            forces[i][X] += f[X];
            forces[i][Y] += f[Y];
        }
        a[i][X] = forces[i][X] / masses[i];
        a[i][Y] = forces[i][Y] / masses[i];
    }

    // calculate new vel
    for (int i = 0; i < n; i++)
    {
        // Calculate a i-th body vel
        vel[i][X] = vel[i][X] + a[i][X] * delta_t;
    }

    // calculate new positions
    for (int i = 0; i < n; i++)
    {
        // Calculate a i-th body position
        
        double t_square = delta_t * delta_t;

        pos[i][X] = pos[i][X] + vel[i][X] * delta_t + a[i][X] * t_square / 2;
        pos[i][Y] = pos[i][Y] + vel[i][Y] * delta_t + a[i][Y] * t_square / 2;
    }
}

int main(int argc, char *argv[])
{
    int n, loc_n;                      // Number of particles, number of particles per process, current particle (loop variable)
    int n_steps;                       // Number of steps, current step
    double delta_t;                    // Step length
    double *masses;                    // Masses of particles
    vect_t *pos, *loc_pos;             // Positions of particles
    vect_t *loc_vel;                   // Speeds of particles in each process
    vect_t *forces, *loc_forces;       // Gravities of particles
    double start, finish;              // Timer
    double parallel_time, serial_time; // parallel and serial duration

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Type_contiguous(DIM, MPI_DOUBLE, &vect_mpi_t); // Derivation type required
    MPI_Type_commit(&vect_mpi_t);

    // Get parameters and initialize arrays
    Get_args(argc, argv, &n, &n_steps, &delta_t);
    loc_n = n / comm_sz; // n % comm_sz == 0
    masses = (double *)malloc(n * sizeof(double));
    pos = (vect_t *)malloc(n * sizeof(vect_t));
    loc_pos = pos + my_rank * loc_n;
    loc_vel = (vect_t *)malloc(loc_n * sizeof(vect_t));
    loc_forces = (vect_t *)malloc(loc_n * sizeof(vect_t));
    if (my_rank == 0)
    {
        vel = (vect_t *)malloc(n * sizeof(vect_t));
        forces = (vect_t *)malloc(n * sizeof(vect_t));
    }
    Gen_init_cond(masses, pos, loc_vel, n, loc_n);

    // Start calculating and timing
    if (my_rank == 0)
        start = MPI_Wtime();

    nbody_parallel(masses, loc_forces, pos, loc_pos, loc_vel, n, loc_n, n_steps, delta_t);

    if (my_rank == 0)
    {
        parallel_time = MPI_Wtime() - start;
    }

    if (my_rank == 0)
    {
        start = MPI_Wtime();
        nbody_serial(masses, forces, pos, vel, n, n_steps, delta_t);
        serial_time = MPI_Wtime() - start;
    }

    // Output time
    if (my_rank == 0)
    {
        printf("serial time = %f s\n", serial_time);
        printf("parallel time = %f s\n", parallel_time);
        free(vel);
        free(forces);
    }
    MPI_Type_free(&vect_mpi_t);
    free(masses);
    free(pos);
    free(loc_vel);
    free(loc_forces);
    MPI_Finalize();
    return 0;
}
