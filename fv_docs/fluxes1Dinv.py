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

    # velocity direction on the cell face
    u_face = (u[:-1] + u[1:])/2.0
    direction = u_face/np.abs(u_face)
    direction = direction.astype(int)
    del u_face

    # find index of upwind cell value
    indxs = np.asarray(range(len(u[:-1])))
    upwind = indxs - (direction - 1)/2
    del indxs
    del direction

    flux = np.empty(len(u)+1)
    flux[1:-1] = u[upwind]*var[upwind]

    # periodic boundary conditions
    u0 = (u[-1] + u[0])/2.0
    if (u0 > 0):
        flux[0] = u[-1]*var[-1]
    if (u0 < 0):
        flux[0] = u[0]*var[0]

    flux[-1] = flux[0]

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

    # average values at cell faces
    u_face = (u[:-1] + u[1:])
    var_face = (var[:-1] + var[1:])

    flux = np.empty(len(u)+1)
    flux[1:-1] = u_face*var_face/4.0

    # periodic boundary conditions
    flux[ 0] = (u[-1] + u[0])*(var[-1] + var[0])/4.0
    flux[-1] = flux[0]

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

    # velocity direction at cell face
    u_face = (u[1:-2] + u[2:-1])/2.0
    direction = u_face/np.abs(u_face)
    direction = direction.astype(int)
    del u_face

    indxs = np.asarray(range(len(u[1:-2])))

    # find indexes for each part of stencil
    upwind   = indxs -  (direction - 1)/2
    upupwind = indxs - ((direction - 1)/2)*3 - 1
    downwind = indxs +  (direction + 1)/2
    del indxs
    del direction

    flux_u  = u[  upwind]*var[  upwind]
    flux_uu = u[upupwind]*var[upupwind]
    flux_d  = u[downwind]*var[downwind]

    flux = np.empty(len(u)+1)
    flux[2:-2] = 0.375*flux_d + 0.75*flux_u - 0.125*flux_uu
    del flux_d
    del flux_u
    del flux_uu

    # periodic boundary conditions
    u_face = (u[-2] + u[-1])/2.0
    if (u_face > 0):
        upup = -3
        up   = -2
        down = -1
    if (u_face < 0):
        upup =  0
        up   = -1
        down = -2
    f_d  = u[down]*var[down]
    f_u  = u[up  ]*var[up  ]
    f_uu = u[upup]*var[upup]
    flux[-2] = 0.375*f_d + 0.75*f_u - 0.125*f_uu

    u_face = (u[-1] + u[0])/2.0
    if (u_face > 0):
        upup = -2
        up   = -1
        down =  0
    if (u_face < 0):
        upup =  1
        upup =  0
        down = -1
    f_d  = u[down]*var[down]
    f_u  = u[up  ]*var[up  ]
    f_uu = u[upup]*var[upup]
    flux[-1] = 0.375*f_d + 0.75*f_u - 0.125*f_uu
    flux[ 0] = flux[-1]

    u_face = (u[0] + u[1])/2.0
    if (u_face > 0):
        upup = -1
        up   =  0
        down =  1
    if (u_face < 0):
        upup = 2
        up   = 1
        down = 0
    f_d  = u[down]*var[down]
    f_u  = u[up  ]*var[up  ]
    f_uu = u[upup]*var[upup]
    flux[1] = 0.375*f_d + 0.75*f_u - 0.125*f_uu

    return flux


