
"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Finite volume numerical viscous flux functions for 1D constant spacing structured meshes
"""

import numpy as np


def CDS2(var, dx, nu):
    """
    CDS2(var, dx, nu)
    var: variable being diffused
    dx: mesh spacing
    nu: diffusion coefficient
    Central Difference Scheme
    Second order accurate
    Estimates diffusive flux at cell face from linear interpolation of neighbouring cell values
    """

    flux = nu*(u[1:] - u[:-1)/dx

    return flux


def CDS4(var, dx, nu):
    """
    CDS4(var, dx, nu)
    var: variable being diffused
    dx: mesh spacing
    nu: diffusion coefficient
    Central Difference Scheme
    Fourth order accurate
    Estimates diffusive flux at cell face from linear interpolation of neighbouring cell values
    """

    indxs = np.asarray(u[1:-2]
    flux = (u[indxs-1] - 27*u[indxs] + 27*u[indxs+1] - u[indxs+2])/24.0
    flux = nu*flux/dx

    return flux

