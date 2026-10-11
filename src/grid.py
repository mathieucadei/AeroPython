import numpy as np
import matplotlib.pyplot as plt

class Grid:

    def __init__(self, x_start, x_end, y_start, y_end, nx, ny):

        self.x = np.linspace(x_start, x_end, nx) 
        self.y = np.linspace(y_start, y_end, ny) 
        self.X, self.Y = np.meshgrid(self.x, self.y)

class PolarGrid:

    def __init__(
            self, 
            radius_start, 
            radius_end, 
            theta_start, 
            theta_end, 
            nradius, 
            ntheta,
            xc,
            yc
        ):

        self.r, self.theta = np.linspace(radius_start, radius_end, nradius), np.linspace(theta_start, theta_end, ntheta)

        r_grid, theta_grid = np.meshgrid(self.r, self.theta)

        z = xc + 1j * yc + r_grid * np.exp(1j * theta_grid)

        self.X, self.Y = z.real, z.imag


if __name__ == '__main__':

    nx, ny = 50, 50                                # number of points in each direction
    x_start, x_end = -2.0, 2.0            # boundaries in the x-direction
    y_start, y_end = -1.0, 1.0            # boundaries in the y-direction

    grid = Grid(x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny)

    plt.scatter(grid.X, grid.Y)
    plt.show()
