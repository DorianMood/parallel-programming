#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cg.h"

// Solve Ax = b for x, using the Conjugate Gradient method.
// Terminates once the maximum number of steps or tolerance has been reached
double *conjugate_gradient_serial(double **A, double *b, int N, int max_steps, double tol)
{

    double *x;

    x = malloc(N * sizeof(double));

    malloc_test(x);

    // Conjugate gradient method implementation

    double *r = b;
    double *ro;

    for (int i = 1; i < max_steps; i++)
    {
        
    }

    return x;
}

void conjugate_gradient_parallel(process_data row, equation_data equation, int N, int max_steps, double tol)
{

    double *x;

    x = &(equation.x_star[0]);

    return;
}
