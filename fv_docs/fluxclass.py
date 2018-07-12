"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Finite volume numerical flux classes for 1D constant spacing structured meshes
"""

import numpy as np
import maths_utils as mth


class Flux(object):
    """
    Parent class for flux types (both inviscid and viscous).
    Has methods for setting boudnary conditions, constructing full flux array, and applying a periodic boundary condition.
    Cannot be used as is, only to be used as parent for specific flux type class.

    Specific flux class must define at least the following additional methods:
    __init__: must at least create an empty list to store boundary conditions (self.bconds), and define the stencil width (self.stencil_radius)
    flux_calculation: applies the normal flux calculation (not at domain boundaries),
    Can also define the following methods, depending on need:

    dirichlet: applies a dirichlet boundary condition at the specified boundary
    neumann: applies a neumann boundary condition at the specified boundary
    robin: applies a robin boundary condition at the specified boundary

    For guidance in writing these methods for specific flux types, please see the templates included in this class, and previously implemented flux types.
    """

    def set_boundary_condition(self, bc_type, bc_index='none'):
        """
        Add a boundary condition of the given type and location to the flux

        input arguments:
        bc_type: must be a string of either 'periodic' (defined for all classes) or other, eg 'dirichlet', 'neumann' (must be written for specific flux type).
        bc_index: if boundary condition is periodic, leave as 'none'. If not, must be either 0 or -1 depending on which boundary the condition applies to

        output arguments:
        none

        If given a boundary condition for an edge which already has one, will replace boundary condition with new one.
        """

        # if new BC is not periodic, must have acceptable boundary index
        # if new BC is periodic, remove all previous BCs
        if (bc_type != 'periodic'):
            if (bc_index == 'none'):
                print('bc_index must be specified for non-periodic bc')
                return
            if ((bc_index != 0) and (bc_index != -1)):
                print('bc_index must be 0 or -1 for non-periodic bc')
                return
        else:
            del self.bconds[:]

        # check if boundary already has boundary condition, and remove if so
        for i, bc in enumerate(self.bconds):
            if ((bc[1] == bc_index) or (bc[0] == 'periodic')):
                del bconds[i]
                break

        # add new BC
        self.bconds.append( (bc_type, bc_index) )

        return

    def apply(self, arglist):
        """
        Construct array of fluxes for all cell faces (including boundary faces)

        input arguments:
        arglist: list of arguments, specific to flux type. Requirements for each flux type are listed in the docstring for that fluxes flux_calculation method

        returns:
        flux: array of fluxes at cell faces. len(flux) = len(var)+1
        """
        flux = np.empty(len(arglist[0])+1)

        # flux in central part of domain
        bound = self.stencil_radius
        flux[bound:-bound] = self.flux_calculation(arglist)

        # flux at boundaries
        for bc in self.bconds:
            bc_type = bc[0]
            bc_indx = bc[1]
            # fetch function for boundary condition
            boundary_condition = getattr(self, bc_type)
            fi_list = boundary_condition(bc_indx, arglist)
            # include boundary fluxes into global flux array
            for f_i in fi_list:
                f = f_i[0]
                i = f_i[1]
                flux[i] = f

        return flux

    def periodic(self, bc_indx, arglist):
        """
        Apply periodic boundary conditions.

        input arguments:
        bc_indx: index of boundary to apply condition to ('none' for periodic)
        arglist: same arglist as is given to 'apply' method (see apply.__doc__ for description)

        returns:
        fi_list: list of tuples. Each tuple contains a flux value (position 0 in tuple) and the index of the global flux array the value corresponds to (position 1 in tuple)

        Periodic plane is the cell face at the beginning/end of the array. First/last CELL FACE is shared, not cell centre value
        """

        # how many cells either side of periodic plane are required to calculate fluxes at these faces
        bound = 2*self.stencil_radius - 1

        # temporary arglist containing arrays of flow variables which span periodic plane, and any other variables from original arglist (eg dx, nu)
        arglist_temp = []
        for arg in arglist:
            if (type(arg) == np.ndarray):
                arg_temp = np.append(arg[-bound:], arg[:bound])
                arglist_temp.append(arg_temp)
            else:
                arglist_temp.append(arg)

        del arglist

        # calculate fluxes across periodic plane consistently with rest of domain
        flux_temp = self.flux_calculation(arglist_temp)

        fi_list = []

        # place flux values into list with relevant global cell face index
        fti = 0
        # fluxes at 'end' of domain
        for fi in range(-self.stencil_radius, -1):
            fi_list.append( (flux_temp[fti], fi) )
            fti += 1
        # shared cell face at periodic plane
        fi_list.append( (flux_temp[fti], -1) )
        fi_list.append( (flux_temp[fti],  0) )
        fti += 1
        # fluxes at 'beginning' of domain
        for fi in range(1, self.stencil_radius):
            fi_list.append( (flux_temp[fti], fi) )
            fti += 1

        return fi_list


class vCDS2(Flux):
    """
    Viscous flux class
    Central Difference Scheme
    Second order
    Stencil: f_{i+1/2} = nu * (u_{i+1} - u_{i}) / dx
    Evaluates viscous flux at cell faces for 1D constant spacing mesh
    """
    def __init__(self):
        """
        initialise flux instance with:
            an empty list for boundary conditions
            the radius of the stencil around the cell face
        """
        self.bconds = []
        self.stencil_radius = 1
        return

    def flux_calculation(self, arglist):
        """
        Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

        input arguments:
        arglist: list with the following elements:
            arglist[0] = var: array of cell centre variable to be diffused
            arglist[1] = dx: distance between cell centres (constant)
            arglist[2] = nu: diffusion coefficient

        returns:
        flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-1
        """
        var = arglist[0]
        dx  = arglist[1]
        nu  = arglist[2]
        del arglist

        flux = var[1:] - var[:-1]
        flux = -nu*flux/dx

        return flux


class vCDS4(Flux):
    """
    Viscous flux class
    Central Difference Scheme
    Second order
    Stencil: f_{i+1/2} = nu * (-u_{i+2} + 27u_{i+1} - 27u_{i} + u_{i-1}) / 24dx
    Evaluates viscous flux at cell faces for 1D constant spacing mesh
    """
    def __init__(self):
        """
        initialise flux instance with:
            an empty list for boundary conditions
            the radius of the stencil around the cell face
        """
        self.bconds = []
        self.stencil_radius = 2
        return

    def flux_calculation(self, arglist):
        """
        Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

        input arguments:
        arglist: list with the following elements:
            arglist[0] = var: array of cell centre variable to be diffused
            arglist[1] = dx: distance between cell centres (constant)
            arglist[2] = nu: diffusion coefficient

        returns:
        flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-3
        """
        var = arglist[0]
        dx  = arglist[1]
        nu  = arglist[2]
        del arglist

        flux = (var[:-3] - 27.0*var[1:-2] + 27.0*var[2:-1] - var[3:]) / 24.0
        flux = -nu*flux/dx

        return flux


class iCDS2(Flux):
    """
    Inviscid flux class
    Central Difference Scheme
    Second order
    Stencil: if f=f(q(x)) f_{i+1/2} = (f(q_{i}) + f(q_{i+1}))/2
    Stencil applied to primitives, flux calculated after
    """
    def __init__(self):
        """
        initialise flux instance with:
            an empty list for boundary conditions
            the radius of the stencil around the cell face
        """
        self.bconds = []
        self.stencil_radius = 1
        return

    def flux_calculation(self, arglist):
        """
        Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

        input arguments:
        arglist: list with the following elements:
            arglist[0] = u: array of cell centre velocities
            arglist[1] = var: array of cell centre variable to be advected

        returns:
        flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-1
        """
        u   = arglist[0]
        var = arglist[1]
        del arglist

        u_cellface   = (u[  :-1] + u[  1:])
        var_cellface = (var[:-1] + var[1:])

        flux = u_cellface*var_cellface/4.0

        return flux


class iCDS2_2(Flux):
    """
    Inviscid flux class
    Central Difference Scheme
    Second order
    Stencil: f_{i+1/2} = (f_{i} + f{i+1})/2
    Stencil applied to fluxes, not primitives
    """
    def __init__(self):
        """
        initialise flux instance with:
            an empty list for boundary conditions
            the radius of the stencil around the cell face
        """
        self.bconds = []
        self.stencil_radius = 1
        return

    def flux_calculation(self, arglist):
        """
        Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

        input arguments:
        arglist: list with the following elements:
            arglist[0] = u: array of cell centre velocities
            arglist[1] = var: array of cell centre variable to be advected

        returns:
        flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-1
        """
        u   = arglist[0]
        var = arglist[1]
        del arglist

        f1 = u[:-1]*var[:-1]
        f2 = u[1: ]*var[1: ]

        flux = (f1 + f2)/2.0

        return flux


class iUDS1(Flux):
    """
    Inviscid flux class
    Upwind Difference Scheme
    First order
    Stencil: f_{i+1/2} = f_{i  }  if u_{i+1/2} > 0
                       = f_{i+1}  if u_{i+1/2} < 0
    """
    def __init__(self):
        """
        initialise flux instance with:
            an empty list for boundary conditions
            the radius of the stencil around the cell face
        """
        self.bconds = []
        self.stencil_radius = 1
        return

    def flux_calculation(self, arglist):
        """
        Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

        input arguments:
        arglist: list with the following elements:
            arglist[0] = u: array of cell centre velocities
            arglist[1] = var: array of cell centre variable to be advected

        returns:
        flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-1
        """
        u   = arglist[0]
        var = arglist[1]
        del arglist

        # velocity direction on cell face for upwinding
        direction = mth.cell_face_direction(u)

        # indexes of cell to left of cell face
        indx_offset = self.stencil_radius - 1
        indxs  = np.asarray(range(len(u[:-1])))
        indxs  = indxs + indx_offset

        # indexes of upwind cells
        upwind = mth.upwind_idx(indxs, direction)
        del indxs
        del direction

        flux = u[upwind]*var[upwind]

        return flux


class iQUICK3(Flux):
    """
    Inviscid flux class
    Quadratic Upwind Interpolation Scheme
    Third order
    Stencil: f_{i+1/2} = 3/4f_{i  } + 3/8f_{i+1} - 1/8f_{i-1}  if u_{i+1/2} > 0
                       = 3/4f_{i+1} + 3/8f_{i  } - 1/8f_{i+2}  if u_{i+1/2} < 0
    """
    def __init__(self):
        """
        initialise flux instance with:
            an empty list for boundary conditions
            the radius of the stencil around the cell face
        """
        self.bconds = []
        self.stencil_radius = 2
        return

    def flux_calculation(self, arglist):
        """
        Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

        input arguments:
        arglist: list with the following elements:
            arglist[0] = u: array of cell centre velocities
            arglist[1] = var: array of cell centre variable to be advected

        returns:
        flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-3
        """
        u   = arglist[0]
        var = arglist[1]
        del arglist

        # velocity direction on cell face for upwinding
        direction = mth.cell_face_direction(u[1:-1])

        # indexes of cell to left of cell face
        indx_offset = self.stencil_radius - 1
        indxs = np.asarray(range(len(u[1:-2])))
        indxs = indxs + indx_offset

        # indexes of up/downwind cells
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

