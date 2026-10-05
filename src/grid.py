import numpy as np
import matplotlib.pyplot as plt

class Grid:

    def __init__(self, x_start, x_end, y_start, y_end, nx, ny):

        self.x = np.linspace(x_start, x_end, nx) 
        self.y = np.linspace(y_start, y_end, ny) 
        self.X, self.Y = np.meshgrid(self.x, self.y)


if __name__ == '__main__':

    nx, ny = 50, 50                                # number of points in each direction
    x_start, x_end = -2.0, 2.0            # boundaries in the x-direction
    y_start, y_end = -1.0, 1.0            # boundaries in the y-direction

    grid = Grid(x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny)

    plt.scatter(grid.X, grid.Y)
    plt.show()