#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cg.h"

// #define DEBUG

void Output_batch(equation_data equation)
{
    printf("%d\n", equation.N);
}

// Euclidian length on given vector
double length(const double *x, const int n)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += pow(x[i], 2);

    return sqrt(sum);
}

// Dot product of given vectors
double vector_dot_product(const double *a, const double *b, const int n)
{
    double product = 0;
    for (int i = 0; i < n; i++)
    {
        product += a[i] * b[i];
    }
    return product;
}

// Dot product of matrix and vector
double *matrix_dot_product(const double *matrix, const double *vector, const int n_rows, const int n_cols)
{
    double *product = (double *)malloc(n_rows * sizeof(double));
    for (int i = 0; i < n_rows; i++)
    {
        product[i] = 0;
        for (int j = 0; j < n_cols; j++)
        {
            product[i] += matrix[i * n_rows + j] * vector[j];
        }
    }
    return product;
}

// Solve Ax = b for x, using the Conjugate Gradient method.
// Terminates once the maximum number of steps or tolerance has been reached
double *conjugate_gradient_serial(const double *A, const double *b, const int N, int max_steps, double tol)
{
    double *x;

    x = malloc(N * sizeof(double));

    malloc_test(x);

    // Conjugate gradient method implementation

    // x = 0
    // r = b
    // rho[0] = length(r)^2
    double *r, *p, *rho;
    p = (double *)malloc(N * sizeof(double));

    // x = 0
    for (int i = 0; i < N; i++)
        x[i] = 0;

    // r = b
    r = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
        r[i] = b[i];

    // rho[0] = length(r) ^ 2
    rho = malloc(N * sizeof(double));
    rho[0] = pow(length(r, N), 2);

    for (int k = 1; k < max_steps; k++)
    {
        // threshhold
        if (sqrt(rho[k - 1]) <= tol * length(b, N))
            break;

        if (k == 1)
        {
            // p = r
            for (int i = 0; i < N; i++)
            {
                p[i] = r[i];
            }
        }
        else
        {
            // p = r + (rho[k - 1] / rho[k - 2]) * p
            for (int i = 0; i < N; i++)
            {
                p[i] = r[i] + (p[k - 1] / p[k - 2]) * p[i];
            }
        }

        // I need dot product:
        // 1. matrix and vecor
        // 2. vector and vector
        double *omega, alpha = 0;
        omega = malloc(N * sizeof(double));

        // omega = A * p;
        omega = matrix_dot_product(A, b, N, N);

        // alpha = p[k - 1] / pT * omega
        alpha = rho[k - 1] / vector_dot_product(p, omega, N);
        // x = x + alpha * p;
        for (int i = 0; i < N; i++)
            x[i] = x[i] + alpha * p[i];
        // r = r - a * omega;
        for (int i = 0; i < N; i++)
            r[i] = r[i] - alpha * omega[i];

        p[k] = pow(length(r, N), 2);
    }

    return x;
}

void conjugate_gradient_parallel(process_data row, equation_data equation, int N, int max_steps, double tol)
{
    double *x;

    x = &(equation.x[0]);

    // Conjugate gradient method implementation

    double *_rows = (double *)malloc(row.N * row.count * sizeof(double));


    // x = 0
    // r = b
    // rho[0] = length(r)^2
    double *r, *p, *rho;
    p = (double *)malloc(row.count * sizeof(double));

    // x = 0
    for (int i = 0; i < row.count; i++)
    {
        x[i] = 0;
    }

    // r = b
    r = malloc(row.count * sizeof(double));
    for (int i = 0; i < row.count; i++)
        r[i] = equation.b[i];

    // rho[0] = length(r) ^ 2
    rho = malloc(row.count * sizeof(double));
    rho[0] = pow(length(r, row.count), 2);

    for (int k = 1; k < max_steps; k++)
    {
        // threshhold
        if (sqrt(rho[k - 1]) <= tol * length(equation.b, row.count))
            break;

        if (k == 1)
        {
            // p = r
            for (int i = 0; i < row.count; i++)
            {
                p[i] = r[i];
            }
        }
        else
        {
            // p = r + (rho[k - 1] / rho[k - 2]) * p
            for (int i = 0; i < row.count; i++)
            {
                p[i] = r[i] + (p[k - 1] / p[k - 2]) * p[i];
            }
        }

        double *omega, alpha = 0;
        omega = malloc(row.count * sizeof(double));

        // omega = A * p;
        for (int i = 0; i < row.count; i++)
        {
            omega[i] = 0;
            for (int j = 0; j < equation.N; j++)
            {
                omega[i] += equation.A[equation.N * i + j] * equation.b[j];
            }
        }
        //omega = matrix_dot_product(equation.A, equation.b, equation.N, equation.N);

        // alpha = rho[k - 1] / pT * omega
        double dot_product = 0;
        for (int i = 0; i < row.count; i++)
        {
            dot_product += p[i] * omega[i];
        }
        alpha = rho[k - 1] / dot_product;
        // x = x + alpha * p;
        for (int i = 0; i < row.count; i++)
            x[i] = x[i] + alpha * p[i];
        // r = r - a * omega;
        for (int i = 0; i < row.count; i++)
            r[i] = r[i] - alpha * omega[i];

        p[k] = pow(length(r, row.count), 2);
    }

    MPI_Barrier(row.comm);

    return;
}
