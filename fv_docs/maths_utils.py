"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Various maths functions
2018/03

    Added:
        L2_norm
        Ln_norm
        line_discretisation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import sympy


def L2_norm(phi, x):
    """
    L2_norm(phi,x)
    phi: function samples
    x  : sample points
    Estimates the Lebesgue 2 norm of a function phi given a set of samples
    at points x. Estimate calculated numerically by Simpsons rule
    using scipy.integrate.simps
    """
    l2n = simps(phi*phi, x)
    return l2n


def Ln_norm(phi, x, n):
    """
    Ln_norm(phi,x,n)
    phi: function samples
    x  : sample points
    n  : order of Lebesgue norm
    Estimates the Lebesgue n norm of a function phi given a set of samples
    at points x. Estimate calculated numerically by Simpsons rule
    using scipy.integrate.simps
    """
    lnn = simps(phi**n, x)
    return lnn


def line_discretisation(length='none', n='none', d='none'):
    """
    line_discretisation(length='none', n='none', d='none')
    length: length of line
    n     : number of discretisation points
    d     : size of discretisation
    Given two of (length, n, d), calculates the third so L=n*d
    If length and d given, and d is not an integer factor of length, calculates
    nearest d which is an integer factor of length, and the corresponding n.
    """
    if length=='none':
        length = n*d
    elif n == 'none':
        n = 1 + int(length/d)
        d = length / (n-1)
    elif d == 'none':
        d = length / (n-1)
    else:
        print('line_discretisation overconstrained. \
                only two inputs must be specified')
    return length, n, d


class SquareWave1D(object):
    """
    Class for square wave with constant convection speed on a 1D periodic domain.
    Must specify initial conditions for max and min wave values, leading and
    trailing edges of wave, convection speed, and domain length.
    """

    def __init__(self):
        pass

    def build(self,
              te_0=0.0,
              le_0=0.5,
              umax=1.0,
              umin=0.0,
              c=1.0,
              L=1.0):

        self.te_0 = te_0
        self.le_0 = le_0
        self.umin = umin
        self.umax = umax
        self.c    = c
        self.L    = L

    def solution(self, x, t):
        """
        returns square wave distribution at time t on grid x
        """

        u = np.zeros(len(x))
        u[:] = self.umin

        le = (self.le_0 + t*self.c) % self.L
        te = (self.te_0 + t*self.c) % self.L

        le_idx = closest_idx(x, le)
        te_idx = closest_idx(x, te)

        if te < le:
            u[te_idx:le_idx] = self.umax
        elif te > le:
            u[:le_idx ] = self.umax
            u[ te_idx:] = self.umax

        return u


def closest_idx(x, target_value):
    """
    closest_idx(x, target_value):
    x: vector of numbers
    target_value: number to find closest x value to
    Returns the index of x with the value closest to target_value
    """
    closest_value_in_x = min(x, key=lambda y: abs(y - target_value))
    idx = np.where(x == closest_value_in_x)[0][0]
    return idx


def BurgersWave1D():
    """
    BurgerWave1D()
    Returns a sympy function u(t, x, nu) corresponding to the exact solution
    to Burgers equation with analytic initial conditions given by Lorena A Barba's CFD Python step 4
    t > 0
    0 < x < 2*pi
    nu > 0
    """
    x, nu, t = sympy.symbols('x nu t')

    phi = (sympy.exp(-(x - 4*t)**2 / (4*nu*(t+1))) + \
           sympy.exp(-(x - 4*t - 2*np.pi)**2 / (4*nu*(t+1))))

    phiprime = phi.diff(x)

    u = -2*nu*(phiprime / phi) + 4

    ufunc = sympy.utilities.lambdify((t, x, nu), u)

    return ufunc


def plot1Dsolution(x, u_n, u_0=None, u_e=None):
    """
    plot1Dsolutions(x, u_n, u_0='none', u_a='none')
    x: array of x axis points
    u_n: numerical solution
    optional arguments
    u_0: initial conditions
    u_e: analytical solution at same time as u_n
    returns a figure with the numerical (and initial/exact if provided)
    solutions plotted and labelled
    """
    fig1, ax1 = plt.subplots(1, 1)

    ax1.plot(x, u_n, label='numerical')
    if (u_0 != None):
        ax1.plot(x, u_0, label='initial')
    if (u_e != None):
        ax1.plot(x, u_e, label='exact')

    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$u$')
    ax1.legend()

    return fig1, ax1


def cell_face_direction(u):
    """
        cell_face_direction(u)
        args:
        u: array len=n of +ve/-ve numbers
        return:
        direction: array of len=n-1
        Values of direction array are +1 or -1 indicating the 'direction' (sign) of the average of consecutive values in u
    """
    direction = np.sign(u[:-1] + u[1:])
    direction = direction.astype(int)
    return direction


def step_into_array( boundary, nsteps ):
    """
    Return index of element nsteps in from array boundary

    input arguments:
    boundary: integer either (0, -1) for start or end of array
    nsteps: how far from array end to step

    return:
    indx: (0+nsteps) or (-1-nsteps)
    """

    direction = boundary*2 + 1
    indx = boundary + nsteps*direction

    return indx


def upwind_idx( indxs, direction, n ):
    upwind = indxs - n*direction + (direction+1)/2
    return upwind


def downwind_idx( indxs, direction, n ):
    downwind = indxs + n*direction - (direction-1)/2
    return downwind


def upwind_idx_obsolete(indxs, direction):
    upwind = indxs - (direction - 1)/2
    return upwind


def upupwind_idx(indxs, direction):
    upupwind = indxs - ((direction - 1)/2)*3 - 1
    return upupwind


def downwind_idx_obsolete(indxs, direction):
    downwind = indxs - (direction + 1)/2
    return downwind


def periodify_sym_patch( array, n ):
    """
    Return array length 2n with last n and first n elements of original array
    """
    return np.append( array[-n:], array[:n] )

def periodify_asym_patch( array, ln, rn ):
    """
    Return array length ln+rn with last ln and first rn elements of original array
    """
    return np.append( array[-ln:], array[:rn] )

def periodify_change_origin( array, origin ):
    """
    Return array rotated periodically until element origin at 0
    """
    ln = len( array ) - origin
    rn = origin
    return self.asym_patch( array, ln, rn )

def periodify_sym_patch_origin( array, n, origin ):
    """
    Return array length 2n with n elements of array from either side of origin from original array
    """
    parray = self.change_origin( array, origin )
    parray = self.sym_patch(     array, n      )
    return parray

def periodify_asym_patch_origin( array, ln, rn, origin ):
    """
    Return array length ln+rn with ln elements from after and rn elements from before origin from original array
    """
    parray = self.change_origin( array, origin )
    parray = self.asym_patch(    array, ln, rn )

    return parray

