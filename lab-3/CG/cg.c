#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cg.h"

// Solve Ax = b for x, using the Conjugate Gradient method.
// Terminates once the maximum number of steps or tolerance has been reached
double *conjugate_gradient_serial(double *A, double *b, int N, int max_steps, double tol) {
  /* PUT OR MODIFY YOUR SERIAL CODE IN THIS FUNCTION*/
  double *x;

  x = malloc(N*sizeof(double));

  malloc_test(x);

  return x;
  /* PUT OR MODIFY YOUR SERIAL CODE IN THIS FUNCTION*/
}


void conjugate_gradient_parallel(process_data row, equation_data equation, int N, int max_steps, double tol) {
  /* PUT OR MODIFY YOUR PARALLEL CODE IN THIS FUNCTION*/
  double *x;
 
  x = &(equation.x_star[0]);

  return;
  /* PUT OR MODIFY YOUR PARALLEL CODE IN THIS FUNCTION*/
}
