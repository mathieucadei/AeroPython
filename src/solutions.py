import numpy as np
import matplotlib.pyplot as plt

from grid import Grid


class SourceSink:

    def __init__(self, x, y, strength):

        self.x = x
        self.y = y
        self.strength = strength

    def velocity_field(self, grid):
        ''' Compute the velocity field on a mesh grid.'''

        u = (self.strength / (2 * np.pi) *
            (grid.X - self.x) / ((grid.X - self.x)**2 + (grid.Y - self.y)**2))
        
        v = (self.strength / (2 * np.pi) *
            (grid.Y - self.y) / ((grid.X - self.x)**2 + (grid.Y - self.y)**2))

        return u, v

    def stream_function(self, grid):

        psi = self.strength / (2 * np.pi) * np.arctan2((grid.Y - self.y), (grid.X - self.x))

        return psi
    
    def phi(self, grid):

        phi = self.strength / (4 * np.pi) * np.log((grid.X - self.x)**2 + (grid.Y - self.y)**2)

        return phi


class UniformFlow:
    
    def __init__(self, U_inf):

        self.U_inf = U_inf

    def velocity_field(self, grid):
        ''' Compute the velocity field on a mesh grid.'''

        u = self.U_inf * np.ones_like(grid.X)
        v = np.zeros_like(grid.Y)

        return u, v

    def stream_function(self, grid):

        psi = self.U_inf * grid.Y

        return psi
    
    def phi(self, grid):

        phi = self.U_inf * grid.X

        return phi


class SourceSinkPair:

    def __init__(self, source, sink):

        self.source = source
        self.sink = sink

    def velocity_field(self, grid):
        ''' Compute the velocity field on a mesh grid.'''

        u_source, v_source = self.source.velocity_field(grid)
        u_sink, v_sink = self.sink.velocity_field(grid)

        u = u_source + u_sink
        v = v_source + v_sink

        return u, v

    def stream_function(self, grid):

        psi = self.source.stream_function(grid) + self.sink.stream_function(grid)

        return psi

    def phi(self, grid):

        phi = self.source.phi(grid) + self.sink.phi(grid)

        return phi


if __name__ == '__main__':

    nx, ny = 50, 50                                # number of points in each direction
    x_start, x_end = -2.0, 2.0            # boundaries in the x-direction
    y_start, y_end = -1.0, 1.0            # boundaries in the y-direction

    grid = Grid(x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny)

    strength_source = 5.0                      # source strength
    x_source, y_source = -1.0, 0.0             # location of the source

    source = SourceSink(strength=strength_source, x=x_source, y=y_source)

    u_source, v_source = source.velocity_field(grid)

    phi_source = source.phi(grid)

    strength_sink = -5.0                      # source strength
    x_sink, y_sink = 1.0, 0.0             # location of the sink

    sink = SourceSink(strength=strength_sink, x=x_sink, y=y_sink)

    phi_sink = sink.phi(grid)

    u_sink, v_sink = sink.velocity_field(grid)

    pair = SourceSinkPair(source=source, sink=sink)

    phi_pair = pair.phi(grid)

    u_pair, v_pair = pair.velocity_field(grid)

    width = 10.0
    height = (y_end - y_start) / (x_end - x_start) * width * 2.0

    fig = plt.figure(figsize=(width, height))

    gs = fig.add_gridspec(3, 1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    ax1.streamplot(grid.X, grid.Y, u_source, v_source, color='0.2',
                    density=2, linewidth=1, arrowsize=2, arrowstyle='->')
    ax1.contour(grid.X, grid.Y, phi_source, colors='k', linewidths=0.5)
    ctr1 = ax1.contourf(grid.X, grid.Y, phi_source, cmap='coolwarm')
    ax1.scatter(x_source, y_source,
                color='g', s=80, marker='o')
    fig.colorbar(ctr1, label='phi')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim(x_start, x_end)
    ax1.set_ylim(y_start, y_end)
    ax1.set_title('Source Streamlines and Phi Contours')

    ax2.streamplot(grid.X, grid.Y, u_sink, v_sink, color='0.2',
                    density=2, linewidth=1, arrowsize=2, arrowstyle='->')
    ax2.contour(grid.X, grid.Y, phi_sink, colors='k', linewidths=0.5)
    ctr2 = ax2.contourf(grid.X, grid.Y, phi_sink, cmap='coolwarm')
    fig.colorbar(ctr2, label='phi')
    ax2.scatter(x_sink, y_sink,
                color='r', s=80, marker='o')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlim(x_start, x_end)
    ax2.set_ylim(y_start, y_end)
    ax2.set_title('Sink Streamlines')

    ax3.streamplot(grid.X, grid.Y, u_pair, v_pair, color='0.2',
                    density=2, linewidth=1, arrowsize=2, arrowstyle='->')
    ax3.contour(grid.X, grid.Y, phi_pair, colors='k', linewidths=0.5)
    ctr3 = ax3.contourf(grid.X, grid.Y, phi_pair, cmap='coolwarm')
    fig.colorbar(ctr3, label='phi')
    ax3.scatter(x_source, y_source,
                color='g', s=80, marker='o')
    ax3.scatter(x_sink, y_sink,
                color='r', s=80, marker='o')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_xlim(x_start, x_end)
    ax3.set_ylim(y_start, y_end)
    ax3.set_title('Source-Sink Pair Streamlines')

    fig.tight_layout()

    plt.show()