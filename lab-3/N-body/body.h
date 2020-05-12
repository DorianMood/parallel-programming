#include <mpi.h>

//#define DEBUG
#define DIM     2                       // 2-d system
#define X       0                       // X coordinate
#define Y       1                       // Y coordinate

typedef double vect_t[DIM];             // Vector data type
MPI_Datatype vect_mpi_t;                // Derived data type used
const double G = 6.67e-11;              // Gravitational constant
int my_rank, comm_sz;                   // Process ID and total number of processes
vect_t *vel = NULL;                     // Global speed, used for outputs of process 0

void Get_args(int argc, char* argv[], int* n_p, int* n_steps_p, double* delta_t_p);
void Gen_init_cond(double masses[], vect_t pos[], vect_t loc_vel[], int n, int loc_n);
void Output_parallel(double masses[], vect_t pos[], vect_t loc_vel[], int n, int loc_n);
void Output_serial(double masses[], vect_t pos[], vect_t vel[], int n);
void nbody_parallel(double masses[], vect_t loc_forces[], vect_t pos[], vect_t loc_pos[], vect_t loc_vel[], int n, int loc_n, int n_steps, double delta_t);
void nbody_serial(double masses[], vect_t forces[], vect_t pos[], vect_t vel[], int n, int n_steps, double delta_t);







