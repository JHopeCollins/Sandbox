"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Finite volume numerical viscous flux functions for 1D constant spacing structured meshes
"""

import numpy as np


def CDS2_func(var, dx, nu):
    """
    CDS2(var, dx, nu)
    var: variable being diffused
    dx: mesh spacing
    nu: diffusion coefficient
    Central Difference Scheme
    Second order accurate
    Estimates diffusive flux at cell face from linear interpolation of neighbouring cell values
    """
    self.flux_lb =  1
    self.flux_up = -1

    flux = np.empty(len(var)+1)
    flux[1:-1] = (var[1:] - var[:-1])

    #periodic boundary conditions
    flux[ 0] = (var[0] - var[-1])
    flux[-1] = flux[0]

    flux = nu*flux/dx

    return flux


def CDS4_func(var, dx, nu):
    """
    CDS4(var, dx, nu)
    var: variable being diffused
    dx: mesh spacing
    nu: diffusion coefficient
    Central Difference Scheme
    Fourth order accurate
    Estimates diffusive flux at cell face from linear interpolation of neighbouring cell values
    """
    self.flux_lb =  2
    self.flux_up = -2
    self.var_lb  =  2
    self.var_ub  = -3

    flux = np.empty(len(var)+1)

    flux[2:-2] = (var[:-3] - 27.0*var[1:-2] + 27.0*var[2:-1] - var[3:]) / 24.0

    # periodic boundary conditions
    flux[-2] = (var[-3] - 27.0*var[-2] + 27.0*var[-1] - var[0]) / 24.0
    flux[ 0] = (var[-2] - 27.0*var[-1] + 27.0*var[ 0] - var[1]) / 24.0
    flux[ 1] = (var[-1] - 27.0*var[ 0] + 27.0*var[ 1] - var[2]) / 24.0

    flux[-1] = flux[0]

    flux = nu*flux/dx

    return flux


class CDS2(object):
    """
    DO NOT USE: NOT FINISHED YET
    Class for CDS2 viscous flux types

    Central Difference Scheme
    Second order accurate
    Estimates diffusive flux at cell face from linear interpolation of neighbouring cell values
    """

    domain_centre = 1

    def flux_calculation(self, arglist):

        var = arglist[0]
        dx  = arglist[1]
        nu  = arglist[2]
        del arglist

        flux = (var[1:] - var[:-1])
        flux = nu*flux/dx

        return flux

    def periodic(self, bc_idx, arglist):

        var = arglist[0]
        dx  = arglist[1]
        nu  = arglist[2]
        del arglist

        var_temp = np.append(var[-1:], var[:0])

        arglist = [var_temp, dx, nu]
        flux_temp = self.flux_calculation(arglist)

        fi_list = []
        fi_list.append( (flux_temp[0], -1) )
        fi_list.append( (flux_temp[0],  0) )

        return fi_list


class CDS4(object):
    """
    DO NOT USE: NOT FINISHED YET
    Class for CDS4 viscous flux types

    Central Difference Scheme
    Fourth order accurate
    Estimates diffusive flux at cell face from cubic interpolation of neighbouring cell values
    """

    domain_centre = 2

    def flux_calculation(self, arglist):

        var = arglist[0]
        dx  = arglist[1]
        nu  = arglist[2]
        del arglist

        flux[2:-2] = (var[:-3] - 27.0*var[1:-2] + 27.0*var[2:-1] - var[3:]) / 24.0
        flux = nu*flux/dx

        return flux

    def periodic(self, bc_idx, arglist):

        var = arglist[0]
        dx  = arglist[1]
        nu  = arglist[2]
        del arglist

        var_temp = np.append(var[-3:], var[:3])

        arglist = [var_temp, dx, nu]

        flux_temp = self.flux_calculation(var_temp, dx, nu)

        fi_list = []
        fi_list.append( (flux_temp[0], -2) )
        fi_list.append( (flux_temp[1], -1) )
        fi_list.append( (flux_temp[1],  0) )
        fi_list.append( (flux_temp[2],  1) )

        return fi_list

