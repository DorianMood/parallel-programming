#include <stdio.h>
#include <omp.h>

void ocean(int **grid, int dim, int timesteps, int threads)
{
    /********************* the red-black algortihm (start)************************/
    /*
    In odd timesteps, calculate indeces with - and in even timesteps, calculate indeces with * 
    See the example of 6x6 matrix, A represents the corner elements. 
        A A A A A A
        A - * - * A
        A * - * - A
        A - * - * A
        A * - * - A
        A A A A A A 
    */

    int step = 0;
    #pragma omp parallel for schedule(dynamic, 2) shared(grid, step, dim)
    for (step = 0; step < timesteps; step++)
    {
        for (int i = 1; i < dim - 1; i++)
        {
            for (int j = 1; j < dim - 1; j++)
            {
                // Even step claculate *
                if ((step % 2) && ((i + j) % 2))
                {
                    grid[j][i] += (grid[i - 1][j] +
                                   grid[i][j - 1] +
                                   grid[i + 1][j] +
                                   grid[i][j + 1]) /
                                  4;
                }// Odd step calculate -
                else if (!(step % 2) && !((i + j) % 2))
                {
                    grid[j][i] += (grid[i - 1][j] +
                                   grid[i][j - 1] +
                                   grid[i + 1][j] +
                                   grid[i][j + 1]) /
                                  4;
                }
            }
            
        }
    }
    /////////////////////// the red-black algortihm (end) ///////////////////////////
}
