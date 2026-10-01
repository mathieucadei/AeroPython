import numpy as np
import matplotlib.pyplot as plt

from grid import Grid
from solutions import SourceSink, UniformFlow, SourceSinkPair


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


class SourceSinkFreestream:

    def __init__(self, source, freestream):

        self.source = source
        self.freestream = freestream

    def velocity_field(self, grid):
        ''' Compute the velocity field on a mesh grid.'''

        u_source, v_source = self.source.velocity_field(grid)
        u_freestream, v_freestream = self.freestream.velocity_field(grid)

        u = u_source + u_freestream
        v = v_source + v_freestream

        return u, v

    def stream_function(self, grid):

        psi_source = self.source.stream_function(grid)
        psi_freestream = self.freestream.stream_function(grid)

        psi = psi_source + psi_freestream

        return psi

    def stagnation_point(self):

        x_stag = self.source.x - self.source.strength / (2.0 * np.pi * self.freestream.U_inf)
        y_stag = self.source.y

        return x_stag, y_stag

    def half_body_max_width(self):

        max_width = self.source.strength / self.freestream.U_inf

        return max_width

    def phi(self, grid):

        phi_source = self.source.phi(grid)
        phi_freestream = self.freestream.phi(grid)

        phi = phi_source + phi_freestream

        return phi      

    def pressure_coefficient(self, grid):

        u, v = self.velocity_field(grid)

        V = np.sqrt(u**2 + v**2)

        cp = 1.0 - (V / self.freestream.U_inf)**2

        return cp

class SourceSinkPairFreestream:

    def __init__(self, source, sink, freestream):

        self.source = source
        self.sink = sink
        self.freestream = freestream

    def velocity_field(self, grid):
        ''' Compute the velocity field on a mesh grid.'''

        u_source, v_source = self.source.velocity_field(grid)
        u_sink, v_sink = self.sink.velocity_field(grid)
        u_freestream, v_freestream = self.freestream.velocity_field(grid)

        u = u_source + u_sink + u_freestream
        v = v_source + v_sink + v_freestream

        return u, v

    def stream_function(self, grid):

        psi_source = self.source.stream_function(grid)
        psi_sink = self.sink.stream_function(grid)
        psi_freestream = self.freestream.stream_function(grid)

        psi = psi_source + psi_sink + psi_freestream

        return psi

    def stagnation_point(self):

        x_source_stag = self.source.x - self.source.strength / (2.0 * np.pi * self.freestream.U_inf)
        y_source_stag = self.source.y

        x_sink_stag = self.sink.x - self.sink.strength / (2.0 * np.pi * self.freestream.U_inf)
        y_sink_stag = self.sink.y

        return x_source_stag, y_source_stag, x_sink_stag, y_sink_stag

    def half_body_max_width(self):

        max_width = 1.0
        for n in range(100):
            max_width_new = self.source.strength / (np.pi * self.freestream.U_inf) * np.arctan(self.source.x / max_width)
            residual = abs(max_width_new - max_width)
            max_width = max_width_new
            if residual < 1e-12:
                break

        return max_width

    def phi(self, grid):

        phi_source = self.source.phi(grid)
        phi_sink = self.sink.phi(grid)
        phi_freestream = self.freestream.phi(grid)

        phi = phi_source + phi_sink + phi_freestream

        return phi      

    def pressure_coefficient(self, grid):

        u, v = self.velocity_field(grid)

        V = np.sqrt(u**2 + v**2)

        cp = 1.0 - (V / self.freestream.U_inf)**2

        return cp


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

    nx, ny = 50, 50                                # number of points in each direction
    x_start, x_end = -4.0, 4.0            # boundaries in the x-direction
    y_start, y_end = -2.0, 2.0            # boundaries in the y-direction

    grid = Grid(x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny)

    strength_source = 5.0                      # source strength
    x_source, y_source = -1.0, 0.0             # location of the source

    source = SourceSink(strength=strength_source, x=x_source, y=y_source)

    u_source, v_source = source.velocity_field(grid)

    phi_source = source.phi(grid)

    u_inf = 1.0

    uniform_flow = UniformFlow(U_inf=u_inf)
    u_freestream, v_freestream = uniform_flow.velocity_field(grid)
    psi_freestream = uniform_flow.stream_function(grid)


    strength_sink = -5.0                      # source strength
    x_sink, y_sink = 1.0, 0.0             # location of the sink

    sink = SourceSink(strength=strength_sink, x=x_sink, y=y_sink)

    phi_sink = sink.phi(grid)

    u_sink, v_sink = sink.velocity_field(grid)

    ranking_half_body = SourceSinkFreestream(source=source, freestream=uniform_flow)

    psi_ranging_half_body = ranking_half_body.stream_function(grid)

    u_ranging_half_body, v_ranging_half_body = ranking_half_body.velocity_field(grid)

    x_ranging_half_body_stag, y_ranging_half_body_stag = ranking_half_body.stagnation_point()

    ranking_half_body_max_width = ranking_half_body.half_body_max_width()

    ranking_oval = SourceSinkPairFreestream(source=source, sink=sink, freestream=uniform_flow)

    psi_ranking_oval = ranking_oval.stream_function(grid)

    u_ranking_oval, v_ranking_oval = ranking_oval.velocity_field(grid)

    x_ranking_oval_source_stag, y_ranking_oval_source_stag, x_ranking_oval_sink_stag, y_ranking_oval_sink_stag = ranking_oval.stagnation_point()

    cp_ranking_oval = ranking_oval.pressure_coefficient(grid)

    ranking_oval_max_width = ranking_oval.half_body_max_width()

    width = 10.0
    height = (y_end - y_start) / (x_end - x_start) * width

    fig1 = plt.figure(figsize=(width, height))

    gs = fig1.add_gridspec(1, 1)

    ax1 = fig1.add_subplot(gs[0, 0])

    ax1.streamplot(grid.X, grid.Y, u_ranging_half_body, v_ranging_half_body,
                    density=2, linewidth=1, arrowsize=2, arrowstyle='->')
    ax1.scatter(x_source, y_source,
                color='g', s=80, marker='o')
    ax1.scatter(x_ranging_half_body_stag, y_ranging_half_body_stag,
                color='r', s=80, marker='o')
    ax1.contour(grid.X, grid.Y, psi_ranging_half_body, 
                levels=[-strength_source / 2.0, strength_source / 2.0],
                colors='r', linewidths=2.0, linestyles='solid')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim(x_start, x_end)
    ax1.set_ylim(y_start, y_end)
    ax1.set_title('Solution Streamlines')

    fig1.tight_layout()

    plt.show()

    width = 15.0
    height = (y_end - y_start) / (x_end - x_start) * width * 2.0

    fig2 = plt.figure(figsize=(width, height), constrained_layout=True)

    gs = fig2.add_gridspec(2, 1)

    ax1 = fig2.add_subplot(gs[0, 0])
    ax2 = fig2.add_subplot(gs[1, 0])

    ax1.streamplot(grid.X, grid.Y, u_ranking_oval, v_ranking_oval,
                    density=2, linewidth=1, arrowsize=2, arrowstyle='->')
    ax1.scatter(x_source, y_source,
                color='g', s=80, marker='o')
    ax1.scatter(x_sink, y_sink,
                color='purple', s=80, marker='o')
    ax1.scatter(x_ranking_oval_source_stag, y_ranking_oval_source_stag,
                color='r', s=80, marker='o')
    ax1.scatter(x_ranking_oval_sink_stag, y_ranking_oval_sink_stag,
                color='r', s=80, marker='o')
    ax1.contour(grid.X, grid.Y, psi_ranking_oval, 
                levels=[0.0],
                colors='r', linewidths=2.0, linestyles='solid')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim(x_start, x_end)
    ax1.set_ylim(y_start, y_end)
    ax1.set_title('Solution Streamlines')

    x_mid = (source.x + sink.x) / 2
    y_top = source.y + ranking_oval_max_width / 2  # full width

    i = np.abs(grid.y - y_top).argmin()
    j = np.abs(grid.x - x_mid).argmin()

    levels = np.linspace(cp_ranking_oval[i, j], cp_ranking_oval.max(), 100)

    ctrf = ax2.contourf(grid.X, grid.Y, cp_ranking_oval, 
                        levels=levels, 
                        cmap='viridis', extend='both')
    ax2.scatter(x_source, y_source,
                color='g', s=80, marker='o')
    ax2.scatter(x_sink, y_sink,
                color='purple', s=80, marker='o')
    ax2.scatter(x_ranking_oval_source_stag, y_ranking_oval_source_stag,
                color='c', s=80, marker='o')
    ax2.scatter(x_ranking_oval_sink_stag, y_ranking_oval_sink_stag,
                color='m', s=80, marker='o')
    ax2.contour(grid.X, grid.Y, psi_ranking_oval, 
                levels=[0.0],
                colors='r', linewidths=2.0, linestyles='solid')
    cbar = fig2.colorbar(ctrf, ax=ax2, label='$C_p$')
    cbar.set_ticks([-2.0, -1.0, 0.0, 1.0])
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlim(x_start, x_end)
    ax2.set_ylim(y_start, y_end)
    ax2.set_title('Solution Pressure Coefficient')

    plt.show()