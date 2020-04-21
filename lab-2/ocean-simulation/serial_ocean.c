#include <stdio.h>

void ocean(int **grid, int dim, int timesteps)
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

    for (int step = 0; step < timesteps; step++)
    {
        for (int i = 1; i < dim - 1; i++)
        {
            for (int j = 1; j < dim - 1; j++)
            {
                if ((step % 2) && ((i + j) % 2))
                {
                    grid[j][i] += (grid[i - 1][j] +
                                   grid[i][j - 1] +
                                   grid[i + 1][j] +
                                   grid[i][j + 1]) /
                                  4;
                }
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
