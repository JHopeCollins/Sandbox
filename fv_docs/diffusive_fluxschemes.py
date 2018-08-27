
"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Finite volume diffusive flux classes for 1D constant spacing structured meshes
"""

import numpy as np
import maths_utils as mth
import fluxclass as flc


class DiffusiveFlux1D( flc.Flux1D ):
    """
        Diffusive flux class for 1D fluxes

        general diffusive flux methods including set diffusion coefficient
    """

    def set_diffusion_coefficient( self, dcoeff ):
        """
            set the diffusion coefficient for the flux
        """
        self.dcoeff = dcoeff
        return

    def construct_arg_list( self ):
        """
        return the argument list for flux_calculation method
        """
        args = []
        args.append( self.var.val )
        args.append( self.mesh.dx )
        args.append( self.dcoeff )
        return args


class CDS2( DiffusiveFlux1D ):
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
        nu  = args[2]

        flux = var[2:-1] - var[1:-2]
        flux = -nu*flux/dx[1:-1]

        return flux


class CDS4( DiffusiveFlux1D ):
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
        nu  = args[2]

        flux = ( var[1:-4] - 27.0*var[2:-3] + 27.0*var[3:-2] - var[4:-1] ) / 24.0
        flux = -nu*flux/dx[2:-2]

        return flux

