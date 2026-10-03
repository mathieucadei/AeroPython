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