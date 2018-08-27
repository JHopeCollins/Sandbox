"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Finite volume numerical flux classes for 1D constant spacing structured meshes
"""

import numpy as np
import maths_utils as mth


class Flux1D( object ):
    """
        Parent class for flux types (both advective and diffusive).

        Has methods for setting boundary conditions, constructing full flux array, and applying a periodic boundary condition.
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
            create Flux instance with variable name and stencil radius
        """
        self.stencil_radius = None
        return

    def set_variable( self, var ):
        """
            Set the variable field that the flux is transporting

            input arguments:
            var: Field1D instance for variable
        """
        self.var  = var
        self.mesh = var.mesh
        return

    def apply( self ):
        """
            Construct array of fluxes for all cell faces (including boundary faces)

            returns:
            flux: array of fluxes at cell faces. len(flux) = len(var)+1
        """
        flux = np.empty( len( self.var.val ) + 1 )

        args = self.construct_arg_list()

        bound = self.stencil_radius + 1

        flux[ bound:-bound ] = self.flux_calculation( args )

        for bc in self.var.bconds:

            bc_func = getattr( self, bc.name )
            fi_list = bc_func( bc, args )

            for f, i in fi_list:
                flux[i] = f

        return flux

    def periodic( self, bc, args ):
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
        bound = 2*self.stencil_radius

        args_temp = []
        for arg in args:
            if type( arg ) == np.ndarray:
                args_temp.append( mth.periodify_sym_patch( arg[1:-1], bound ) )
            else:
                args_temp.append( arg )

        flux_temp = self.flux_calculation( args_temp )

        fi_list = []

        # place flux values into list with relevant global cell face index
        # ti: temp flux index; i: global flux index
        ti = 0

        for i in range( -self.stencil_radius-1, -2 ):
            fi_list.append( (flux_temp[ti], i) )
            ti += 1

        fi_list.append( (flux_temp[ti], -2) )
        fi_list.append( (flux_temp[ti],  1) )
        ti += 1

        for i in range( 2, self.stencil_radius+1 ):
            fi_list.append( (flux_temp[ti], i) )
            ti += 1

        return fi_list

