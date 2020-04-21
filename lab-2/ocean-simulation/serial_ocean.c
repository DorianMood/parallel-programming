#include <stdio.h>

void ocean (int **grid, int dim, int timesteps)
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

    // PUT YOUR CODE HERE

    for (int step = 0; step < timesteps; step++)
    {
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                if ((i + j) % 2)
                {
                    grid[j][i] += ((step - 1) / 2) % 2 ? 1 : -1;
                }
                else
                {
                    grid[j][i] += (step / 2) % 2 ? 1 : -1;                    
                }
            }
        }
    }

    /////////////////////// the red-black algortihm (end) ///////////////////////////
}
