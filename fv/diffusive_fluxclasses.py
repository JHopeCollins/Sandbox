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
        self.dcoeff = dcoeff
        return

    def arg_list( self, q ):
        """
        return the argument list for flux_calculation method
        """
        args = []
        args.append( q.val )
        args.append( q.mesh.dxp )
        args.append( q.mesh.dxh )
        args.append( self.dcoeff )
        return args


