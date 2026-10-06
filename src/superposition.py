import numpy as np
import matplotlib.pyplot as plt

from src.grid import Grid
from src.elementary_solutions import SourceSink, Doublet, Vortex, UniformFlow


class Superposition:

    def __init__(self, elementary_solutions):

        self.elementary_solutions = elementary_solutions

    def velocity_field(self, grid):

        u_total = np.zeros_like(grid.X)
        v_total = np.zeros_like(grid.Y)

        for solution in self.elementary_solutions:
            u, v = solution.velocity_field(grid)
            u_total += u
            v_total += v

        return u_total, v_total

    def stream_function(self, grid):

        psi_total = np.zeros_like(grid.X)

        for solution in self.elementary_solutions:
            psi = solution.stream_function(grid)
            psi_total += psi

        return psi_total

    def phi(self, grid):

        phi_total = np.zeros_like(grid.X)

        for solution in self.elementary_solutions:
            phi = solution.phi(grid)
            phi_total += phi

        return phi_total

    def pressure_coefficient(self, grid, u_inf):

        u_total, v_total = self.velocity_field(grid)

        V = np.sqrt(u_total**2 + v_total**2)

        cp = 1.0 - (V / u_inf)**2

        return cp

    def max_pressure_coefficient(self, cp):

        cp_max = np.max(cp)

        cp_max_indice = np.argmax(cp)

        cp_max_row, cp_max_column = np.unravel_index(cp_max_indice, cp.shape)

        return cp_max, cp_max_indice, cp_max_row, cp_max_column


class RankineHalfBody(Superposition):

    def __init__(self, source, freestream):

        super().__init__(elementary_solutions=[source, freestream])

        self.source = source
        self.freestream = freestream

    def pressure_coefficient(self, grid):

        return super().pressure_coefficient(grid, self.freestream.U_inf)
    
    def stagnation_point(self):

        x_stag = self.source.x - self.source.strength / (2.0 * np.pi * self.freestream.U_inf)
        y_stag = self.source.y

        return x_stag, y_stag

    def half_body_max_width(self):

        max_width = self.source.strength / self.freestream.U_inf

        return max_width
    

class RankineOval(Superposition):

    def __init__(self, source, sink, freestream):

        super().__init__(elementary_solutions=[source, sink, freestream])

        self.source = source
        self.sink = sink
        self.freestream = freestream

    def pressure_coefficient(self, grid):

        return super().pressure_coefficient(grid, self.freestream.U_inf)

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


class DoubletFreestream(Superposition):

    def __init__(self, doublet, freestream):

        super().__init__(elementary_solutions=[doublet, freestream])

        self.doublet = doublet
        self.freestream = freestream

    def pressure_coefficient(self, grid):

        return super().pressure_coefficient(grid, self.freestream.U_inf)

    def stagnation_point(self):

        x_stagn1, y_stagn1 = np.sqrt(self.doublet.strength / (2.0 * np.pi * self.freestream.U_inf)), 0.0
        x_stagn2, y_stagn2 = -np.sqrt(self.doublet.strength / (2.0 * np.pi * self.freestream.U_inf)), 0.0

        return x_stagn1, y_stagn1, x_stagn2, y_stagn2

    def radius(self):

        return np.sqrt(self.doublet.strength / (2.0 * np.pi * self.freestream.U_inf))

    def surface_speed(self, theta):

        return 2.0 * self.freestream.U_inf * np.abs(np.sin(theta))

    def pressure_distribution(self, theta):

        return 1.0 - 4.0 * np.sin(theta)**2

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

    pair = Superposition(elementary_solutions=[source, sink])

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

    ranking_half_body = RankineHalfBody(source=source, freestream=uniform_flow)

    psi_ranging_half_body = ranking_half_body.stream_function(grid)

    u_ranging_half_body, v_ranging_half_body = ranking_half_body.velocity_field(grid)

    x_ranging_half_body_stag, y_ranging_half_body_stag = ranking_half_body.stagnation_point()

    ranking_half_body_max_width = ranking_half_body.half_body_max_width()

    ranking_oval = RankineOval(source=source, sink=sink, freestream=uniform_flow)

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


    nx, ny = 50, 50                                # number of points in each direction
    x_start, x_end = -2.0, 2.0            # boundaries in the x-direction
    y_start, y_end = -1.0, 1.0            # boundaries in the y-direction

    grid = Grid(x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny)

    kappa = 1.0
    x_doublet, y_doublet = 0.0, 0.0

    doublet = Doublet(xd=x_doublet, yd=y_doublet, strength=kappa)
    u_doublet, v_doublet = doublet.velocity_field(grid)
    psi_doublet = doublet.stream_function(grid)

    width = 10
    height = (y_end - y_start) / (x_end - x_start) * width

    fig, ax = plt.subplots(figsize=(width, height))

    ax.streamplot(grid.X, grid.Y, u_doublet, v_doublet,
                  density=2, linewidth=1, arrowsize=1, arrowstyle='->')
    ax.scatter(x_doublet, y_doublet, color='r', s=80, marker='o')
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16) 
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(y_start, y_end)
    plt.show()

    doublet_freestream = DoubletFreestream(doublet=doublet, freestream=uniform_flow)

    u_doublet_freestream, v_doublet_freestream = doublet_freestream.velocity_field(grid)  
    psi_doublet_freestream = doublet_freestream.stream_function(grid)

    x_stag1, y_stag1, x_stag2, y_stag2 = doublet_freestream.stagnation_point()

    cp_doublet_freestream = doublet_freestream.pressure_coefficient(grid)

    doublet_freestreem_radius = doublet_freestream.radius()

    theta = np.linspace(0, 2.0 * np.pi, 200)

    doublet_freestreem_surface_speed = doublet_freestream.surface_speed(theta)

    doublet_freestreem_pressure_distribution = doublet_freestream.pressure_distribution(theta)

    # Surface coordinates and speed from your object
    x_surface = x_doublet + doublet_freestreem_radius * np.cos(theta)
    y_surface = y_doublet + doublet_freestreem_radius * np.sin(theta)

    width = 10
    height = 10

    fig, ax = plt.subplots(2, 1, figsize=(width, height), layout='compressed')

    ax[0].streamplot(grid.X, grid.Y, u_doublet_freestream, v_doublet_freestream,
                     density=2, linewidth=1, arrowsize=1, arrowstyle='->')
    ax[0].contour(grid.X, grid.Y, psi_doublet_freestream,
                  levels=[0.0], colors='r', linewidths=2.0, linestyles='solid') 
    ax[0].scatter(x_doublet, y_doublet, color='r', s=80, marker='o')
    ax[0].scatter([x_stag1, x_stag2], [y_stag1, y_stag2], color='g', s=80, marker='o')

    surface = ax[0].scatter(
        x_surface, y_surface,
        c=doublet_freestreem_surface_speed,
        cmap='plasma',
        vmin=0,
        vmax=2.0 * doublet_freestream.freestream.U_inf,
        s=20,
        zorder=5,
    )

    fig.colorbar(surface, ax=ax[0], label="Surface speed")

    # # Keep the cylinder circular in both plots
    # for axis in ax:
    #     axis.set_aspect('equal', adjustable='box')

    ax[0].set_xlabel('x', fontsize=16)
    ax[0].set_ylabel('y', fontsize=16)
    ax[0].set_xlim(x_start, x_end)
    ax[0].set_ylim(y_start, y_end)

    contf = ax[1].contourf(grid.X, grid.Y, cp_doublet_freestream, levels=np.linspace(-3.0, 1.0, 100), cmap='viridis', extend='both')
    cbar = fig.colorbar(contf)
    cbar.set_label('$C_p$', fontsize=16)
    cbar.set_ticks([-2.0, -1.0, 0.0, 1.0])
    ax[1].scatter(x_doublet, y_doublet, color='r', s=80, marker='o')
    ax[1].contour(grid.X, grid.Y, psi_doublet_freestream,
                  levels=[0.0], colors='r', linewidths=2.0, linestyles='solid')
    ax[1].scatter([x_stag1, x_stag2], [y_stag1, y_stag2], color='g', s=80, marker='o')

    surface = ax[1].scatter(
        x_surface, y_surface,
        c=doublet_freestreem_pressure_distribution,
        cmap='viridis',
        vmin=-3.0, 
        vmax=1.0,
        s=20,
        zorder=5,
    )

    ax[1].set_xlabel('x', fontsize=16)
    ax[1].set_ylabel('y', fontsize=16)
    ax[1].set_xlim(x_start, x_end)
    ax[1].set_ylim(y_start, y_end)

    for axis in ax:
        axis.set_aspect('equal', adjustable='box')

    plt.show()


    gamma = 5.0
    x_vortex, y_vortex = 0.0, 0.0

    vortex = Vortex(strength=gamma, x=x_vortex, y=y_vortex)

    u_vortex, v_vortex = vortex.velocity_field(grid)
    psi_vortex = vortex.stream_function(grid)

    width = 10
    height = (y_end - y_start) / (x_end - x_start) * width
    fig, ax = plt.subplots(figsize=(width, height))
    ax.streamplot(grid.X, grid.Y, u_vortex, v_vortex,
                    density=2, linewidth=1, arrowsize=1, arrowstyle='->')
    ax.scatter(x_vortex, y_vortex, color='r', s=80, marker='o')
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(y_start, y_end)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Vortex Streamlines')
    plt.show()

    sigma_sink = -1.0
    x_sink, y_sink = 0.0, 0.0

    sink = SourceSink(strength=sigma_sink, x=x_sink, y=y_sink)

    vortex_sink_pair = Superposition(elementary_solutions=[vortex, sink])

    u_vortex_sink_pair, v_vortex_sink_pair = vortex_sink_pair.velocity_field(grid)

    psi_vortex_sink_pair = vortex_sink_pair.stream_function(grid)

    width = 10
    height = (y_end - y_start) / (x_end - x_start) * width
    fig, ax = plt.subplots(figsize=(width, height))
    ax.streamplot(grid.X, grid.Y, u_vortex_sink_pair, v_vortex_sink_pair,
                    density=2, linewidth=1, arrowsize=1, arrowstyle='->')
    ax.scatter(x_vortex, y_vortex, color='r', s=80, marker='o')
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(y_start, y_end)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Vortex-Sink Streamlines')
    plt.show()