
"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Finite volume advective flux classes for 1D constant spacing structured meshes
"""

import numpy as np
import maths_utils as mth
import fluxclass as flc


class AdvectiveFlux1D( flc.Flux1D ):
    """
        Advective flux class for 1D fluxes

        general advective flux methods including set advection velocity
    """
    def set_advection_velocity( self, v ):
        self.vel = v
        return

    def arg_list( self, q ):
        """
        return the argument list for flux_calculation() method
        """
        args = []
        args.append( q.val )
        args.append( q.mesh.dxp )
        args.append( q.mesh.dxh )
        args.append( self.vel.val )
        return args


class REAFlux1D( AdvectiveFlux1D ):
    """
        Reconstruct, Evolve, Average flux scheme for 1D fluxes

        subclasses each define a numerical flux from the cell face jump. requires a reconstruction method to calculate this jump
    """

    def set_reconstruction_radius( self, r ):
        self.stencil_radius = r
        return

    def set_reconstruction( self, r ):
        self.reconstruct = r
        return

    def set_evolution( self, e ):
        self.evolve = e
        return

    def flux_calculation( self, args ):
        q   = args[0]
        dxp = args[1]
        dxh = args[2]
        v   = args[3]

        qL, qR = self.reconstruct( q, dxp, dxh )
        vL, vR = self.reconstruct( v, dxp, dxh )

        flux = self.evolve( qL,
                            qR,
                            vL,
                            vR)

        return flux


class UpwindFlux1D( AdvectiveFlux1D ):
    """
        Upwind advective flux class for 1D fluxes

        general upwind methods including upwind/downwind index calculation
    """

    def cell_face_direction( self, u ):
        """
        return sign of average value of u at cell faces
        """
        r = self.stencil_radius
        if r==1:
            return np.sign( u[   :-r] + u[r:      ] ).astype( int )
        return     np.sign( u[r-1:-r] + u[r:-(r-1)] ).astype( int )

    def cell_indxs( self, u ):
        """
        return indices of cell to left of each domain-centre face
        """
        r = self.stencil_radius
        l = len( u ) +1 - 2*r
        return np.asarray( range( l ) )

    def upwind_indx( self, indxs, direction, n ):
        return indxs - n*direction + (direction+1)/2

    def downwind_indx( self, indxs, direction, n ):
        return indxs + n*direction - (direction-1)/2


class PressureFlux1D( flc.Flux1D ):
    def set_pressure( self, p ):
        self.p = p
        return

    def arg_list( self, q ):
        args = []
        args.append( q.val )
        args.append( q.mesh.dxp )
        args.append( q.mesh.dxh )
        args.append( self.p.val )
        return args


class REAPressureFlux1D(  PressureFlux1D ):
    def set_reconstruction_radius( self, r ):
        self.stencil_radius = r
        return

    def set_reconstruction( self, r ):
        self.reconstruct = r
        return

    def set_evolution( self, e ):
        self.evolve = e
        return

    def flux_calculation( self, args ):
        r = self.stencil_radius
        q   = args[0]
        dxp = args[1]
        dxh = args[2]
        p   = args[3]
        v   = args[4]
        v   = np.ones( len(v) )

        pL, pR = self.reconstruct( p[1:-1], dxp[1:-1], dxh[1:-1] )
        vL, vR = self.reconstruct( v[1:-1], dxp[1:-1], dxh[1:-1] )

        flux = self.evolve( pL,
                            pR,
                            vL,
                            vR)

        return flux
       

