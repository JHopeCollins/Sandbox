"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Finite volume numerical flux classes for 1D constant spacing structured meshes
"""

import numpy as np
import maths_utils as mth


class Flux1D( object ):
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

    def __init__( self ):
        """
        create Flux instance
        creates the following (empty) attributes:
            string: variable_name
            list:   other_variable_names
            list:   parameter_names
            list:   bconds
            int:    stencil_radius
        """
        self.variable_name = None
        self.other_variable_names = []
        self.parameter_names = []
        self.bconds = []
        self.stencil_radius = None
        self.implemented_bconds = [ 'periodic' ]

    def set_variable( self, variable_name ):
        """
        Set the name of the variable that the flux is transporting

        input arguments:
        variable_name: string containing variable name. MUST BE KEY FOR VARIABLE DICTIONARY
        """
        self.variable_name = variable_name
        return

    def set_other_variables( self, other_variable_names ):
        """
        Set the names of other variables required to calculate the flux

        input arguments:
        other_variable_name: list of strings containing other variable names. MUST BE KEYS FOR VARIABLE DICTIONARY

        NB order of variable names must match that required by specific flux calculation
        """
        self.other_variable_names = other_variable_names
        return

    def set_parameters( self, parameter_names ):
        """
        Set the parameters required to calculate the flux

        input arguments:
        parameter_names: list of strings containing parameter names. MUST BE KEYS FOR PARAMETER DICTIONARY KEYS

        NB order of parameter names must match that required by specific flux calculation
        """
        self.parameter_names = parameter_names
        return

    def construct_arg_lists( self, var_dict, par_dict ):
        """
        Return the variable, and lists of the other variables and parameters required by the flux

        input arguments:
        var_dict: dictionary containing all flow variables, with their names as keys
        par_dict: dictionary containing all flow parameters, with their names as keys

        output:
        var_list: list of flow variables required by the flux, in the order expected
        par_list: list of flow parameters required by the flux, in the order expected
        """
        var = var_dict[ self.variable_name ]
        other_var_list = []
        par_list = []
        for name in self.other_variable_names:
            other_var_list.append( var_dict[name] )
        for name in self.parameter_names:
            par_list.append( par_dict[name] )

        return var, other_var_list, par_list

    def set_boundary_condition( self, bc_type, bc_index=None, bc_value=None ):
        """
        Add a boundary condition of the given type and location to the flux

        input arguments:
        bc_type: must be a string of either 'periodic' (defined for in Flux class) or other, eg 'dirichlet', 'neumann' (check if written for specific flux class).
        bc_index: if boundary condition is periodic, leave as None. If not, must be either 0 or -1 depending on which boundary the condition applies to
        bc_val: if boundary condition is periodic, leave as None. If not, must be value required by condition, eg variable value for dirichlet

        output arguments:
        none

        If given a boundary condition for an edge which already has one, will replace boundary condition with new one.
        """

        if bc_type not in self.implemented_bconds:
            print( 'bc_type %s not implemented for this flux' % bc_type)
            return

        if bc_type == 'periodic':

            del self.bconds[:]
            self.bconds.append( ( bc_type, bc_index, bc_value ) )
            return

        else:

            if bc_index == None:
                print( 'bc_index must be specified for non-periodic bc_type %s' % bc_type )
                return
            if ( bc_index != 0 ) and ( bc_index != -1 ):
                print( 'bc_index must be 0 or -1 for non-periodic boundary condition on 1D flux for bc_type %s' % bc_type )
                return

            # check if boundary already has boundary condition, and remove if so
            for i, bc in enumerate(self.bconds):
                if ( bc[1] == bc_index ) or ( bc[0] == 'periodic' ):
                    del self.bconds[i]
                    break

            self.bconds.append( ( bc_type, bc_index, bc_value ) )

        return

    def apply( self, var_dict, par_dict ):
        """
        Construct array of fluxes for all cell faces (including boundary faces)

        input arguments:
        var_dict: dictionary containing all flow variables, with their names as keys
        par_dict: dictionary containing all parameters, with their names as keys

        returns:
        flux: array of fluxes at cell faces. len(flux) = len(var)+1
        """
        flux = np.empty( par_dict['nx'] + 1 )

        var, other_var_list, par_list = self.construct_arg_lists( var_dict, par_dict )

        bound = self.stencil_radius

        flux[ bound:-bound ] = self.flux_calculation( var, other_var_list, par_list )

        for bc_type, bc_indx, bc_val in self.bcond:

            boundary_condition = getattr( self, bc_type )
            fi_list = boundary_condition( bc_indx, bc_val, var, other_var_list, par_list )

            for f, i in fi_list:
                flux[i] = f

        return flux

    def periodic( self, bc_indx, bc_val, var, other_var_list, par_list ):
        """
        Apply periodic boundary conditions.

        input arguments:
        bc_indx: index of boundary to apply condition to (None for periodic)
        bc_val: value boundary condition set to (None for periodic)
        var: array of the variable being transported by the flux
        other_var_list: list of arrays of other flow variables needed for the flux calculation
        par_list: list of parameters needed for the flux calculation

        returns:
        fi_list: list of tuples. Each tuple contains a flux value (position 0 in tuple) and the index of the global flux array the value corresponds to (position 1 in tuple)

        Periodic plane is the cell face at the beginning/end of the array. First/last CELL FACE is shared, not cell centre value
        """

        # how many cells either side of periodic plane are required to calculate fluxes at these faces
        bound = 2*self.stencil_radius - 1

        var_temp = np.append( var[-bound:], var[:bound] )
        other_var_list_temp = []
        par_list_temp = []

        for other_var in other_var_list:
            if type( other_var ) == np.ndarray:
                other_var_temp = np.append( other_var[-bound:], other_var[:bound] )
                other_var_list_temp.append( other_var_temp )
            else:
                other_var_list_temp.append( other_var )

        for par in par_list:
            if type( par ) == np.ndarray:
                par_temp = np.append( par[-bound:], par[:bound] )
                par_list_temp.append( par_temp )
            else:
                par_list_temp.append( par )

        flux_temp = self.flux_calculation( var_temp, other_var_list_temp, par_list_temp )

        fi_list = []

        # place flux values into list with relevant global cell face index
        # ti: temp flux index; i: global flux index
        ti = 0

        for i in range( -self.stencil_radius, -1 ):
            fi_list.append( (flux_temp[ti], i) )
            ti += 1

        fi_list.append( (flux_temp[ti], -1) )
        fi_list.append( (flux_temp[ti],  0) )
        ti += 1

        for i in range( 1, self.stencil_radius ):
            fi_list.append( (flux_temp[ti], i) )
            ti += 1

        return fi_list


class vCDS2( Flux1D ):
    """
    Diffusive flux class
    Central Difference Scheme
    Second order
    Stencil: f_{i+1/2} = nu * (u_{i+1} - u_{i}) / dx
    Evaluates diffusive flux at cell faces for 1D constant spacing mesh

    Required keys in namelists:
    other_variable_names:
        None
    parameter_names:
        cell spacing;
        diffusion coefficient;
    """
    def __init__( self ):
        """
        initialise flux instance with Flux class attributes and the radius of the stencil around the cell face
        """
        super( self.__class__, self ).__init__()
        self.stencil_radius = 1
        return

    def flux_calculation( self, var, other_var_list, par_list ):
        """
        Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

        input arguments:
        var: variable to be transported
        other_var_list: empty
        par_list: [ cell spacing , diffusion coefficient ]

        returns:
        flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-1
        """
        dx  = par_list[0]
        nu  = par_list[1]

        flux = var[1:] - var[:-1]
        flux = -nu*flux/dx

        return flux


class vCDS4( Flux1D ):
    """
    Diffusive flux class
    Central Difference Scheme
    Fourth order
    Stencil: f_{i+1/2} = nu * (-u_{i+2} + 27u_{i+1} - 27u_{i} + u_{i-1}) / 24dx
    Evaluates diffusive flux at cell faces for 1D constant spacing mesh

    Required keys in namelists (in order):
    other_variable_names:
        None
    parameter_names:
        cell spacing;
        diffusion coefficient
    """
    def __init__( self ):
        """
        initialise flux instance with Flux class attributes and the radius of the stencil around the cell face
        """
        super( self.__class__, self ).__init__()
        self.stencil_radius = 2
        return

    def flux_calculation( self, var, other_var_list, par_list ):
        """
        Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

        input arguments:
        var: variable to be transported
        other_var_list: empty
        par_list: [ cell spacing , diffusion coefficient ]

        returns:
        flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-3
        """
        dx  = par_list[0]
        nu  = par_list[1]

        flux = ( var[:-3] - 27.0*var[1:-2] + 27.0*var[2:-1] - var[3:] ) / 24.0
        flux = -nu*flux/dx

        return flux


class iCDS2( Flux1D ):
    """
    Advective flux class
    Central Difference Scheme
    Second order
    Stencil: if f=f(q(x)) f_{i+1/2} = (f(q_{i}) + f(q_{i+1}))/2
    Calculate flux from reconstructed primitives

    Required keys in namelists:
    other_variable_names:
        Advection velocity
    parameter_names:
        None
    """
    def __init__( self ):
        """
        initialise flux instance with Flux class attributes and the radius of the stencil around the cell face
        """
        super( self.__class__, self ).__init__()
        self.stencil_radius = 1
        self.implemented_bconds.append( 'dirichlet' )
        return

    def flux_calculation( self, var, other_var_list, par_list ):
        """
        Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

        input arguments:
        var: variable to be transported
        other_var_list: [ advection velocity ]
        par_list: None

        returns:
        flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-1
        """
        u = other_var_list[0]

        u_cellface   = ( u[  :-1] + u[  1:] )
        var_cellface = ( var[:-1] + var[1:] )

        flux = u_cellface*var_cellface/4.0

        return flux

    def dirichlet( self, bc_indx, bc_val, var, other_var_list, par_list ):
        """
        Apply dirichlet boundary conditions.

        input arguments:
        bc_indx: index of boundary to apply condition to
        bc_val: value of var at boundary
        var: array of the variable being transported by the flux
        other_var_list: list of arrays of other flow variables needed for the flux calculation
        par_list: list of parameters needed for the flux calculation

        returns:
        fi_list: list of tuples. Each tuple contains a flux value (position 0 in tuple) and the index of the global flux array the value corresponds to (position 1 in tuple)
        """

        u = other_var_list[0]

        indx_in = mth.step_into_array( bc_indx, 1 )

        u_boundary   = u[bc_indx] - ( u[indx_in] - u[bc_indx] ) / 2.0
        var_boundary = bc_val

        f = u_boundary*var_boundary

        fi_list = [ ( f, bc_indx ) ]

        return fi_list


class iCDS2_2( Flux1D ):
    """
    Advective flux class
    Central Difference Scheme
    Second order
    Stencil: f_{i+1/2} = (f_{i} + f{i+1})/2
    Stencil applied to fluxes, not primitives
    Reconstruct flux from calculate flux distribution

    Required keys in namelists:
    other_variable_names:
        Advection velocity
    parameter_names:
        None
    """
    def __init__( self ):
        """
        initialise flux instance with Flux class attributes and the radius of the stencil around the cell face
        """
        super( self.__class__, self ).__init__()
        self.stencil_radius = 1
        return

    def flux_calculation( self, var, other_var_list, par_list ):
        """
        Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

        input arguments:
        var: variable to be transported
        other_var_list: [ advection velocity ]
        par_list: None

        returns:
        flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-1
        """
        u = other_var_list[0]

        f1 = u[:-1]*var[:-1]
        f2 = u[1: ]*var[1: ]

        flux = ( f1 + f2 )/2.0

        return flux


class iUDS1( Flux1D ):
    """
    Advective flux class
    Upwind Difference Scheme
    First order
    Stencil: f_{i+1/2} = f_{i  }  if u_{i+1/2} > 0
                       = f_{i+1}  if u_{i+1/2} < 0

    Required keys in namelists:
    other_variable_names:
        Advection velocity
    parameter_names:
        None
    """
    def __init__( self ):
        """
        initialise flux instance with Flux class attributes and the radius of the stencil around the cell face
        """
        super( self.__class__, self ).__init__()
        self.stencil_radius = 1
        self.implemented_bconds.append( 'dirichlet' )
        return

    def flux_calculation( self, var, other_var_list, par_list ):
        """
        Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

        input arguments:
        var: variable to be transported
        other_var_list: [ advection velocity ]
        par_list: None

        returns:
        flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-1
        """
        u = other_var_list[0]

        direction = mth.cell_face_direction( u )

        # indexes of cell to left of cell face
        indx_offset = self.stencil_radius - 1
        indxs  = np.asarray( range( len( u[:-1] ) ) )
        indxs  = indxs + indx_offset

        # indexes of upwind cells
        upwind = mth.upwind_idx( indxs, direction )
        del indxs
        del direction

        flux = u[upwind]*var[upwind]

        return flux

    def dirichlet( self, bc_indx, bc_val, var, other_var_list, par_list ):
        """
        Apply dirichlet boundary conditions.

        input arguments:
        bc_indx: index of boundary to apply condition to
        bc_val: value of var at boundary
        var: array of the variable being transported by the flux
        other_var_list: list of arrays of other flow variables needed for the flux calculation
        par_list: list of parameters needed for the flux calculation

        returns:
        fi_list: list of tuples. Each tuple contains a flux value (position 0 in tuple) and the index of the global flux array the value corresponds to (position 1 in tuple)
        """

        # | 0 | 1 | 2  >  -3  | -2  | -1  |
        # 0 . 1 . 2 .  >   . -3  . -2  . -1

        u = other_var_list[0]

        u_boundary   = u[bc_indx]
        var_boundary = bc_val

        f = u[bc_indx]*bc_val

        fi_list = [ ( f, bc_indx ) ]

        return fi_list


class iQUICK3( Flux1D ):
    """
    Advective flux class
    Quadratic Upwind Interpolation Scheme
    Third order
    Stencil: f_{i+1/2} = 3/4f_{i  } + 3/8f_{i+1} - 1/8f_{i-1}  if u_{i+1/2} > 0
                       = 3/4f_{i+1} + 3/8f_{i  } - 1/8f_{i+2}  if u_{i+1/2} < 0

    Required keys in namelists:
    other_variable_names:
        Advection velocity
    parameter_names:
        None
    """
    def __init__( self ):
        """
        initialise flux instance with Flux class attributes and the radius of the stencil around the cell face
        """
        super( self.__class__, self ).__init__()
        self.stencil_radius = 2
        return

    def flux_calculation( self, var, other_var_list, par_list ):
        """
        Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

        input arguments:
        var: variable to be transported
        other_var_list: [ advection velocity ]
        par_list: None

        returns:
        flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-3
        """
        u   = other_var_list[0]

        # velocity direction on cell face for upwinding
        direction = mth.cell_face_direction( u[1:-1] )

        # indexes of cell to left of cell face
        indx_offset = self.stencil_radius - 1
        indxs = np.asarray( range( len( u[1:-2]) ) )
        indxs = indxs + indx_offset

        # indexes of up/downwind cells
        upwind   = mth.upwind_idx(   indxs, direction )
        upupwind = mth.upupwind_idx( indxs, direction )
        downwind = mth.downwind_idx( indxs, direction )
        del indxs
        del direction

        flux_u  = u[   upwind ]*var[   upwind ]
        flux_uu = u[ upupwind ]*var[ upupwind ]
        flux_d  = u[ downwind ]*var[ downwind ]

        flux = 0.375*flux_d + 0.75*flux_u - 0.125*flux_uu

        return flux

