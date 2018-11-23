"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Finite volume diffusive flux classes for 1D constant spacing structured meshes
"""

import numpy as np
import maths_utils as mth
import diffusive_fluxclasses as dfc


class CDS2( dfc.DiffusiveFlux1D ):
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
        super( CDS2, self ).__init__()
        self.stencil_radius = 1
        return

    def flux_calculation( self, args ):
        """
            Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

            input arguments:
            var: variable to be transported
            other_var_list: empty
            par_list: [ cell spacing , diffusion coefficient ]

            returns:
            flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-1
        """
        var = args[0]
        dx  = args[1]
        h   = args[2]
        nu  = args[3]

        flux = var[1:] - var[:-1]
        flux = -nu*flux/dx[:]

        return flux


class CDS4( dfc.DiffusiveFlux1D ):
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
        super( CDS4, self ).__init__()
        self.stencil_radius = 2
        return

    def flux_calculation( self, args ):
        """
            Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

            input arguments:
            var: variable to be transported
            other_var_list: empty
            par_list: [ cell spacing , diffusion coefficient ]

            returns:
            flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-3
        """
        var = args[0]
        dx  = args[1]
        h   = args[2]
        nu  = args[3]

        flux = ( var[:-3] - 27.0*var[1:-2] + 27.0*var[2:-1] - var[3:] ) / 24.0
        flux = -nu*flux/dx[1:-1]

        return flux

