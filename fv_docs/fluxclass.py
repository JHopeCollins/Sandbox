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
        self.stencil_radius = 1
        return

    def set_mesh( self, x ):
        """
            Set the variable field that the flux is transporting

            input arguments:
            var: Field1D instance for variable
        """
        self.mesh = x
        return

    def apply( self, q ):
        """
            Construct array of fluxes for all cell faces (excluding faces outside ghosts)

            returns:
            flux: array of fluxes at cell faces. len(flux) = len(var)+1
        """
        flux = np.zeros( len( q.val_noghost ) + 1 )

        args = self.arg_list( q )

        bound = self.stencil_radius

        flux[ bound:-bound ] = self.flux_calculation( args )

        for bc in q.bconds:
            bc_func = getattr( self, bc.name )
            bc_func( bc, flux, args )

        return flux

    def periodic( self, bc, flux, args ):
        """
            Apply periodic boundary conditions.

            input arguments:
            bc: BoundaryCondition instance
            args: arguments for flux calculation

            returns:
            fi_list: list of tuples. Each tuple contains a flux value (position 0 in tuple) and the index of the global flux array the value corresponds to (position 1 in tuple)

            Periodic plane is the cell face at the beginning/end of the array. First/last CELL FACE is shared, not cell centre value
        """
        bound = 2*self.stencil_radius

        args_temp = []
        for arg in args:
            if arg is self.mesh.dx:
                args_temp.append( mth.periodify_sym_patch( arg[1:-1], bound ) )

                pdx = ( self.mesh.dx[0] + self.mesh.dx[-1] ) /2.0
                args_temp[-1] = args_temp[-1][1:-1]
                args_temp[-1] = np.insert( args_temp[-1], bound-1, pdx )

            elif type( arg ) == np.ndarray:
                args_temp.append( mth.periodify_sym_patch( arg[1:-1], bound ) )

            else:
                args_temp.append( arg )

        flux_temp = self.flux_calculation( args_temp )

        fi_list = []

        ti = 0
        for i in range( -self.stencil_radius, -1 ):
            flux[i] = flux_temp[ti]
            ti += 1

        flux[-1] = flux_temp[ti]
        flux[ 0] = flux_temp[ti]
        ti += 1

        for i in range( 1, self.stencil_radius ):
            flux[i] = flux_temp[ti]
            ti += 1

        return

    def arg_list( self, q ):
        args = []
        args.append( q.val )
        args.append( self.mesh.dx )
        args.append( self.mesh.h  )
        return args

    def flux_calculation( self, args ):
        var = args[0]
        return np.asarray( range( len( var ) -3 ) ) +1

