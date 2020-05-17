#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cg.h"

double length(const double *x, const int n)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += pow(x[i], 2);
        
    return sqrt(sum);
}

// Solve Ax = b for x, using the Conjugate Gradient method.
// Terminates once the maximum number of steps or tolerance has been reached
double *conjugate_gradient_serial(const double *A, const double *b, const int N, int max_steps, double tol)
{

    double *x;

    x = malloc(N * sizeof(double));

    malloc_test(x);

    // Conjugate gradient method implementation

    double *r, *p;

    r = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
        r[i] = b[i];
    
    p = malloc(N * sizeof(double));
    p[0] = pow(length(r, N), 2);

    for (int k = 1; k < max_steps; k++)
    {
        if (p[k - 1] <= length(b, N))
            break;

        if (k == 1)
            for (int i = 0; i < N; i++)
                p[i] = r[i];
        else
            for (int i = 0; i < N; i++)
                p[i] = r[i] + (p[k - 1] / p[k - 2]) * p[i];
        
        double *omega, *alpha;
        omega = malloc(N * sizeof(double));
        alpha = malloc(N * sizeof(double));
        
        // omega = A * p;
        for (int i = 0; i < N; i++)
            omega[i] = A[i] * p[i];
        // alpha = p[k - 1] / pT * omega
        for (int i = 0; i < N; i++)
            alpha[i] = p[k - 1] / p[i] * omega[i];
        // x = x + alpha * p;
        for (int i = 0; i < N; i++)
            x[i] = x[i] + alpha[i] * p[i];
        // r = r - a * omega;
        for (int i = 0; i < N; i++)
            r[i] = r[i] - alpha[i] * omega[i];

        p[k] = pow(length(r, N), 2);
    }

    return x;
}

void conjugate_gradient_parallel(process_data row, equation_data equation, int N, int max_steps, double tol)
{

    double *x;

    x = &(equation.x_star[0]);

    return;
}
