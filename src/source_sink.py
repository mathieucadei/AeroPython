import numpy as np
import matplotlib.pyplot as plt

from grid import Grid


class SourceSink:

    def __init__(self, x, y, strength):

        self.x = x
        self.y = y
        self.strength = strength

    def velocity_at(self, X, Y):
        ''' Compute the velocity field on a mesh grid.'''

        u = (self.strength / (2 * np.pi) *
            (X - self.x) / ((X - self.x)**2 + (Y - self.y)**2))
        
        v = (self.strength / (2 * np.pi) *
            (Y - self.y) / ((X - self.x)**2 + (Y - self.y)**2))

        return u, v

    def phi(self, X, Y):

        phi = self.strength / (4 * np.pi) * np.log((X - self.x)**2 + (Y - self.y)**2)

        return phi


class SourceSinkPair:

    def __init__(self, source, sink):

        self.source = source
        self.sink = sink

    def velocity_at(self, X, Y):
        ''' Compute the velocity field on a mesh grid.'''

        u_source, v_source = self.source.velocity_at(X, Y)
        u_sink, v_sink = self.sink.velocity_at(X, Y)

        u = u_source + u_sink
        v = v_source + v_sink

        return u, v

    def phi(self, X, Y):

        phi = self.source.phi(X, Y) + self.sink.phi(X, Y)

        return phi


if __name__ == '__main__':

    nx, ny = 50, 50                                # number of points in each direction
    x_start, x_end = -2.0, 2.0            # boundaries in the x-direction
    y_start, y_end = -1.0, 1.0            # boundaries in the y-direction

    grid = Grid(x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny)

    strength_source = 5.0                      # source strength
    x_source, y_source = -1.0, 0.0             # location of the source

    source = SourceSink(strength=strength_source, x=x_source, y=y_source)

    u_source, v_source = source.velocity_at(grid.X, grid.Y)

    phi_source = source.phi(grid.X, grid.Y)

    strength_sink = -5.0                      # source strength
    x_sink, y_sink = 1.0, 0.0             # location of the sink

    sink = SourceSink(strength=strength_sink, x=x_sink, y=y_sink)

    phi_sink = sink.phi(grid.X, grid.Y)

    u_sink, v_sink = sink.velocity_at(grid.X, grid.Y)

    pair = SourceSinkPair(source=source, sink=sink)

    phi_pair = pair.phi(grid.X, grid.Y)

    u_pair, v_pair = pair.velocity_at(grid.X, grid.Y)

    width = 15.0
    height = (y_end - y_start) / (x_end - x_start) * width * 1.5

    fig = plt.figure(figsize=(width, height))

    gs = fig.add_gridspec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    ax1.streamplot(grid.X, grid.Y, u_source, v_source,
                    density=2, linewidth=1, arrowsize=2, arrowstyle='->')
    ax1.scatter(x_source, y_source,
                color='g', s=80, marker='o')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim(x_start, x_end)
    ax1.set_ylim(y_start, y_end)
    ax1.set_title('Source Psi Streamlines')

    ax2.contour(grid.X, grid.Y, phi_source, colors='k', linewidths=0.5)
    ctr1 = ax2.contourf(grid.X, grid.Y, phi_source)
    fig.colorbar(ctr1, label='phi')
    ax2.scatter(x_source, y_source,
                color='g', s=80, marker='o')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlim(x_start, x_end)
    ax2.set_ylim(y_start, y_end)
    ax2.set_title('Source Phi Contours')

    ax3.streamplot(grid.X, grid.Y, u_sink, v_sink,
                    density=2, linewidth=1, arrowsize=2, arrowstyle='->')
    ax3.scatter(x_sink, y_sink,
                color='r', s=80, marker='o')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_xlim(x_start, x_end)
    ax3.set_ylim(y_start, y_end)
    ax3.set_title('Sink Psi Streamlines')

    ax4.contour(grid.X, grid.Y, phi_sink, colors='k', linewidths=0.5)
    ctr2 = ax4.contourf(grid.X, grid.Y, phi_sink)
    fig.colorbar(ctr2, label='phi')
    ax4.scatter(x_sink, y_sink,
                color='r', s=80, marker='o')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_xlim(x_start, x_end)
    ax4.set_ylim(y_start, y_end)
    ax4.set_title('Sink Phi Contours')

    ax5.streamplot(grid.X, grid.Y, u_pair, v_pair,
                    density=2, linewidth=1, arrowsize=2, arrowstyle='->')
    ax5.scatter(x_source, y_source,
                color='g', s=80, marker='o')
    ax5.scatter(x_sink, y_sink,
                color='r', s=80, marker='o')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_xlim(x_start, x_end)
    ax5.set_ylim(y_start, y_end)
    ax5.set_title('Source-Sink Pair Psi Streamlines')

    ax6.contour(grid.X, grid.Y, phi_pair, colors='k', linewidths=0.5)
    ctr3 = ax6.contourf(grid.X, grid.Y, phi_pair)
    fig.colorbar(ctr3, label='phi')
    ax6.scatter(x_source, y_source,
                color='g', s=80, marker='o')
    ax6.scatter(x_sink, y_sink,
                color='r', s=80, marker='o')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_xlim(x_start, x_end)
    ax6.set_ylim(y_start, y_end)
    ax6.set_title('Source-Sink Pair Phi Contours')

    fig.tight_layout()

    plt.show()