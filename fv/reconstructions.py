"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Reconstruction functions for REA type finite volume numerical schemes
Each function takes in a single array of cell average values and returns left and right face values for each cell
"""

import numpy as np
import maths_utils as mth


class Reconstruction1D( object ):
    def __init__( self ):
        super( Reconstruction1D, self ).__init__()
        self.stencil_radius = None
        return


class SecondOrderReconstruction1D( Reconstruction1D ):
    def __init__( self ):
        super( Reconstruction1D, self ).__init__()
        self.stencil_radius = 2
        return

    def dirichlet_ghosts( self, bc, q ):
        qb = np.zeros( 2 )

        inside  = bc.indx +1
        outside = bc.indx

        qb[  inside ] = q[ bc.indx ]
        qb[ outside ] = 2.0*bc.val - q[ bc.indx ]
        return qb

    def neumann_ghosts( self, bc, q ):
        qb = np.zeros( 2 )
        n   = 2*bc.indx +1

        inside  = bc.indx +1
        outside = bc.indx

        dx = q.mesh.dxp[ bc.indx ]
        dn = bc.val

        ghost = q[ bc.indx ] - n*dn*dx

        qb[  inside ] = q[ bc.indx ]
        qb[ outside ] = ghost
        return qb


class PCM1( Reconstruction1D ):
    def __init__( self ):
        super( PCM1, self ).__init__()
        self.stencil_radius = 1
        return

    def reconstruct( self, q, dx, h ):
        """
        returns left and right cell face values for piecewise constant reconstruction
        """
        qR = q[1:]
        qL = q[:-1]
        return qL, qR

    def dirichlet_ghosts( self, bc, q ):
        qb = np.zeros( 2 )

        inside  = bc.indx +1
        outside = bc.indx

        qb[  inside ] = q[ bc.indx ]
        qb[ outside ] = bc.val
        return qb

    def neumann_ghosts( self, bc, q ):
        qb = np.zeros( 2 )
        n   = 2*bc.indx +1

        inside  = bc.indx +1
        outside = bc.indx

        dx = q.mesh.dxp[ bc.indx ]
        dn = bc.val

        ghost = q[ bc.indx ] - n*dn*dx

        qb[  inside ] = q[ bc.indx ]
        qb[ outside ] = ghost
        return qb


class minmod2( SecondOrderReconstruction1D ):
    def reconstruct( self, q, dx, h ):
        """
        returns left and right cell face values for piecewise linear reconstruction with minmod slope choice

        minmod chooses slope to be minimum of neighbouring slopes, or zero if cell at minimum/maximum (when slopes of opposite sign)
        """

        sigmaL = ( q[1:-1] - q[ :-2] ) / dx[ :-1]
        sigmaR = ( q[2:  ] - q[1:-1] ) / dx[1:  ]

        sigma  = mth.minmod( sigmaL, sigmaR )

        dq = 0.5*sigma*h[1:-1]
        qR = q[ 2:-1 ] - dq[1:  ]
        qL = q[ 1:-2 ] + dq[ :-1]

        return qL, qR

    def dirichlet_ghosts( self, bc, q ):
        qb = np.zeros( 5 )
        n   = 2*bc.indx +1

        # internal cells
        ixi = bc.indx + 2*n
        iqi = bc.indx
        for i in range( 3 ):
            qb[ ixi ] = q[ iqi ]
            ixi += n
            iqi += n

        # ghost cells
        ghost0 = 3.0*q[ bc.indx   ] - 2.0*bc.val
        sigmaR = n*( q[ bc.indx+n ] - q[ bc.indx ] )
        sigmaL = n*( q[ bc.indx   ] - ghost0 )

        if abs( sigmaR ) < abs( sigmaL ):
            ghost0 = bc.val
            ghost1 = bc.val
        else:
            ghost1 = 3.0*ghost0 - 2.0*bc.val

        qb[ bc.indx   ] = ghost1
        qb[ bc.indx+n ] = ghost0

        return qb


class superbee2( SecondOrderReconstruction1D ):
    def reconstruct( self, q, dx, h ):
        """
        returns left and right cell face values for piecewise linear reconstruction with superbee slope choice 
        
        superbee chooses slope to be minimum of each one-sided slope compared with twice the other one-sided slope, or zero if cell at minimum/maximum (when slopes of opposite sign)
        This choice of slopes tends to choose the larger slope in smooth regions, but still the smaller slope near discontinuities
        """
    
        sigmaL = ( q[1:-1] - q[ :-2] ) / dx[ :-1]
        sigmaR = ( q[2:  ] - q[1:-1] ) / dx[1:  ]
    
        sigma1  = mth.minmod(   sigmaL, 2*sigmaR )
        sigma2  = mth.minmod( 2*sigmaL,   sigmaR )
    
        sigma = mth.maxmod( sigma1, sigma2 )
    
        dq = 0.5*sigma*h[1:-1]
        qR = q[ 2:-1 ] - dq[1:  ]
        qL = q[ 1:-2 ] + dq[ :-1]
    
        return qL, qR


class MC2( SecondOrderReconstruction1D ):
    def reconstruct( self, q, dx, h ):
        """
        returns left and right cell face values for piecewise linear reconstruction with monotized central-difference limiter choice of slopes

        MC chooses slope to be minimum of twice each one-sided slope or the central difference slope, or zero if cell at minimum/maximum (when slopes of opposite sign)
        This choice of slopes tends to choose the central slope in smooth regions, which results in better resolution than the superbee choice, which tends to artificially steepen smooth regions
        """
        sigmaL = ( q[1:-1] - q[ :-2] ) / dx[1:  ]
        sigmaR = ( q[2:  ] - q[1:-1] ) / dx[ :-1]
        sigmaC = ( q[2:  ] - q[ :-2] ) / ( dx[1:] + dx[:-1] )

        sigma = mth.minmod3( 2*sigmaL, 2*sigmaR, sigmaC )

        dq = 0.5*sigma*h[1:-1]
        qR = q[ 2:-1 ] - dq[1:  ]
        qL = q[ 1:-2 ] + dq[ :-1]

        return qL, qR

