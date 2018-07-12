"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Finite volume numerical flux classes for 1D constant spacing structured meshes
"""

import numpy as np
import maths_utils as mth


class Flux(object):
    """
    DO NOT USE: NOT FINISHED YET
    Class for CDS2 viscous flux types
    """

    # def __init__(self):
    #   self.bconds = []

    def set_boundary_condition(self, bc_type, bc_index='none'):

        if (bc_type != 'periodic'):
            if (bc_index == 'none'):
                print('bc_index must be specified for non-periodic bc')
                return
            if ((bc_index != 0) and (bc_index != -1)):
                print('bc_index must be 0 or -1 for non-periodic bc')
                return

        # check if boundary already has boundary condition, remove if so
        for i, bc in enumerate(self.bconds):
            if (bc[1] == bc_index):
                del bconds[i]
                break

        self.bconds.append( (bc_type, bc_index) )

        return

    def apply(self, arglist):

        flux = np.empty(len(arglist[0])+1)

        # flux in central part of domain
        dc = self.domain_centre
        flux[dc:-dc] = self.flux_calculation(arglist)

        # flux at boundaries
        for bc in self.bconds:
            bc_type = bc[0]
            bc_indx = bc[1]
            # list of fluxes and their intended index
            fi_list = getattr(self, bc_type)(bc_indx, arglist)
            for f_i in fi_list:
                f = f_i[0]
                i = f_i[1]
                flux[i] = f

        return flux


class vCDS2(Flux):

    def __init__(self):
        self.bconds = []
        self.domain_centre = 1
        return

    def flux_calculation(self, arglist):

        var = arglist[0]
        dx  = arglist[1]
        nu  = arglist[2]
        del arglist

        flux = var[1:] - var[:-1]
        flux = -nu*flux/dx

        return flux

    def periodic(self, bc_idx, arglist):

        var = arglist[0]
        dx  = arglist[1]
        nu  = arglist[2]
        del arglist

        var_temp = np.append(var[-1:], var[:1])

        arglist = [var_temp, dx, nu]
        flux_temp = self.flux_calculation(arglist)

        fi_list = []
        fi_list.append( (flux_temp[0], -1) )
        fi_list.append( (flux_temp[0],  0) )

        return fi_list


class vCDS4(Flux):

    def __init__(self):

        self.bconds = []
        self.domain_centre = 2
        return

    def flux_calculation(self, arglist):

        var = arglist[0]
        dx  = arglist[1]
        nu  = arglist[2]
        del arglist

        flux = (var[:-3] - 27.0*var[1:-2] + 27.0*var[2:-1] - var[3:]) / 24.0
        flux = -nu*flux/dx

        return flux

    def periodic(self, bc_idx, arglist):

        var = arglist[0]
        dx  = arglist[1]
        nu  = arglist[2]
        del arglist

        var_temp = np.append(var[-3:], var[:3])

        arglist = [var_temp, dx, nu]

        flux_temp = self.flux_calculation(arglist)

        fi_list = []
        fi_list.append( (flux_temp[0], -2) )
        fi_list.append( (flux_temp[1], -1) )
        fi_list.append( (flux_temp[1],  0) )
        fi_list.append( (flux_temp[2],  1) )

        return fi_list


class iCDS2(Flux):

    def __init__(self):

        self.bconds = []
        self.domain_centre = 1
        return

    def flux_calculation(self, arglist):

        u   = arglist[0]
        var = arglist[1]
        del arglist

        u_face   = (u[  :-1] + u[  1:])
        var_face = (var[:-1] + var[1:])

        flux = u_face*var_face/4.0

        return flux

    def periodic(self, bc_indx, arglist):

        u   = arglist[0]
        var = arglist[1]
        del arglist

        u_temp   = np.append(u[  -1:], u[  :1])
        var_temp = np.append(var[-1:], var[:1])

        arglist = [u_temp, var_temp]
        flux_temp = self.flux_calculation(arglist)

        fi_list = []
        fi_list.append( (flux_temp[0], -1) )
        fi_list.append( (flux_temp[0],  0) )

        return fi_list


class iUSD1(Flux):

    def __init__(self):
        self.bconds = []
        self.domain_centre = 1
        return

    def flux_calculation(self, arglist):

        u   = arglist[0]
        var = arglist[1]
        del arglist

        # velocity direction on cell face for upwinding
        direction = mth.cell_face_direction(u)

        # indexes of upwind  cells
        indxs  = np.asarray(range(len(u[:-1])))
        upwind = mth.upwind_idx(indxs, direction)
        del indxs
        del direction

        flux = u[upwind]*var[upwind]

        return flux

    def periodic(self, bc_indx, arglist):

        u   = arglist[0]
        var = arglist[1]
        del arglist

        u_temp    = np.append(u[  -1:], u[  :1])
        var_temp  = np.append(var[-1:], var[:1])

        arglist = [u_temp, var_temp]
        flux_temp = self.flux_calculation(arglist)

        fi_list = []
        fi_list.append( (flux_temp[0], -1) )
        fi_list.append( (flux_temp[0],  0) )

        return  fi_list


class iQUICK3(Flux):

    def __init__(self):

        self.bconds = []
        self.domain_centre = 2
        return

    def flux_calculation(self, arglist):

        u   = arglist[0]
        var = arglist[1]
        del arglist

        # velocity direction on cell face for upwinding
        direction = mth.cell_face_direction(u[1:-1])

        # indexes of upwind cells
        indxs    = np.asarray(range(len(u[1:-2])))
        upwind   = mth.upwind_idx(  indxs, direction)
        upupwind = mth.upupwind_idx(indxs, direction)
        downwind = mth.downwind_idx(indxs, direction)
        del indxs
        del direction

        flux_u  = u[  upwind]*var[  upwind]
        flux_uu = u[upupwind]*var[upupwind]
        flux_d  = u[downwind]*var[downwind]

        flux = 0.375*flux_d + 0.75*flux_u - 0.125*flux_uu

        return flux

    def periodic(self, bc_indx, arglist):
        """
        Some issues here 11/07/18
        Produces blip at boundary
        """

        u   = arglist[0]
        var = arglist[1]
        del arglist

        u_temp    = np.append(u[  -3:], u[  :3])
        var_temp  = np.append(var[-3:], var[:3])

        arglist = [u_temp, var_temp]
        flux_temp = self.flux_calculation(arglist)

        fi_list = []
        fi_list.append( (flux_temp[0], -2) )
        fi_list.append( (flux_temp[1], -1) )
        fi_list.append( (flux_temp[1],  0) )
        fi_list.append( (flux_temp[2],  1) )

        return  fi_list


