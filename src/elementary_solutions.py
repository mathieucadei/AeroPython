import numpy as np
import matplotlib.pyplot as plt


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


class Doublet:

    def __init__(self, xd, yd, strength):

        self.xd = xd
        self.yd = yd
        self.strength = strength

    def velocity_field(self, grid):
        ''' Compute the velocity field on a mesh grid.'''

        u = (-self.strength / (2 * np.pi) *
            ((grid.X - self.xd)**2 - (grid.Y - self.yd)**2) /
            ((grid.X - self.xd)**2 + (grid.Y - self.yd)**2)**2)
        
        v = (-self.strength / (2 * np.pi) *
            (2 * (grid.X - self.xd) * (grid.Y - self.yd)) /
            ((grid.X - self.xd)**2 + (grid.Y - self.yd)**2)**2)

        return u, v

    def stream_function(self, grid):

        psi = (-self.strength / (2 * np.pi) *
            (grid.Y - self.yd) / ((grid.X - self.xd)**2 + (grid.Y - self.yd)**2))

        return psi
    
    def phi(self, grid):

        phi = (-self.strength / (2 * np.pi) *
            (grid.X - self.xd) / ((grid.X - self.xd)**2 + (grid.Y - self.yd)**2))

        return phi


class Vortex:
    
    def __init__(self, x, y, gamma):

        self.x = x
        self.y = y
        self.gamma = gamma

    def velocity_field(self, grid):
        ''' Compute the velocity field on a mesh grid.'''

        u = (self.gamma / (2 * np.pi) *
            (grid.Y - self.y) / ((grid.X - self.x)**2 + (grid.Y - self.y)**2))
        
        v = (-self.gamma / (2 * np.pi) *
            (grid.X - self.x) / ((grid.X - self.x)**2 + (grid.Y - self.y)**2))

        return u, v

    def stream_function(self, grid):

        psi = (self.gamma / (4 * np.pi) *
            np.log((grid.X - self.x)**2 + (grid.Y - self.y)**2))

        return psi
    
    def phi(self, grid):

        phi = (self.gamma / (2 * np.pi) *
            np.arctan2((grid.Y - self.y), (grid.X - self.x)))

        return phi

class InfiniteRowOfVortices:

    def __init__(self, gamma, spacing):

        self.gamma = gamma
        self.spacing = spacing

    def velocity_field(self, grid):
        ''' Compute the velocity field on a mesh grid.'''

        u = self.gamma / (2 * self.spacing) * np.sinh(2 * np.pi * grid.Y / self.spacing) / \
            (np.cosh(2 * np.pi * grid.Y / self.spacing) - np.cos(2 * np.pi * grid.X / self.spacing))

        v = - self.gamma / (2 * self.spacing) * np.sin(2 * np.pi * grid.X / self.spacing) / \
            (np.cosh(2 * np.pi * grid.Y / self.spacing) - np.cos(2 * np.pi * grid.X / self.spacing))

        return u, v
    
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