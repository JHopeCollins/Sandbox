"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Finite volume numerical inviscid flux functions for 1D constant spacing structured meshes
"""

import numpy as np

def UDS1(u, var):
    """
    UDS2(u, var)
    u: velocity vector
    var: variable being convected
    Upwind Difference Scheme
    First order accurate
    Estimated inviscid flux at cell face from upwind cell values
    """

    u_face = (u[:-1] + u[1:])/2.0
    direction = int(u_face/np.abs(u_face))
    del u_face

    indxs = np.asarray(range(len(u[:-1])))
    upwind = indxs - (direction - 1)/2
    del indxs

    flux = u[upwind]*var[upwind]

    return flux


def CDS2(u, var):
    """
    CDS2(u, var)
    u: velocity vector
    var: variable being convected
    Central Difference Scheme
    Second order accurate
    Estimates inviscid flux at cell face from linear interpolation of neighbouring cell values
    """

    u_face = (u[:-1] + u[1:])/2.0
    var_face = (var[:-1] + var[1:])/2.0

    flux = u_face*var_face

    return flux


def QUICK3(u, var):
    """
    QUICK(u, var):
    u: velocity vector
    var: variable being convected
    Quadratic Upwind Interpolation
    Third order accurate
    Estimate inviscid flux at cell face from semi-upstream quadratic interpolation
    """

    u_face = (u[:-2] + u[1:-1])/2.0
    direction = int(u_face/np.abs(u_face))
    del u_face

    indxs = np.asarray(range(len(u[1:-2])))

    upwind   = indxs -  (direction - 1)/2
    upupwind = indxs - ((direction - 1)/2)*3 - 1
    downwind = indxs +  (direction + 1)/2
    del indxs
    del direction

    flux_u  = u[  upwind]*var[  upwind]
    flux_uu = u[upupwind]*var[upupwind]
    flux_d  = u[downwind]*var[downwind]

    flux = 0.375*flux_d + 0.75*flux_u - 0.125*flux_uu

    return flux



