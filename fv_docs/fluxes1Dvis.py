
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

    flux = np.empty(len(var)+1)
    flux[1:-1] = (var[1:] - var[:-1])

    #periodic boundary conditions
    flux[ 0] = (var[0] - var[-1])
    flux[-1] = flux[0]

    flux = nu*flux/dx

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

    flux = np.empty(len(var)+1)

    flux[2:-2] = (var[:-3] - 27.0*var[1:-2] + 27.0*var[2:-1] - var[3:]) / 24.0

    # periodic boundary conditions
    flux[-2] = (var[-3] - 27.0*var[-2] + 27.0*var[-1] - var[0]) / 24.0
    flux[ 0] = (var[-2] - 27.0*var[-1] + 27.0*var[ 0] - var[1]) / 24.0
    flux[ 1] = (var[-1] - 27.0*var[ 0] + 27.0*var[ 1] - var[2]) / 24.0

    flux[-1] = flux[0]

    flux = nu*flux/dx

    return flux

