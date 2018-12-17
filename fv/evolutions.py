"""
Written by J Hope-Collins (jth39@cam.ac.uk)

Numerical flux functions for REA type finite volume schemes
Each function takes arrays of left and right values of advection velocity and advected quantity at each face, and returns the numerical flux associated with this jump
"""

import numpy as np

class Evolution1D( object ):
    def __init__( self ):
        super( Evolution1D, self ).__init__()


class UpwindEvolution1D( Evolution1D ):
    def direction(self,  vL, vR ):
        d = np.sign( vL + vR ).astype( int )
        return d


class Upwind1( UpwindEvolution1D ):
    def evolve( self, qL, qR, vL, vR ):
        flux = np.zeros_like( qL )

        direction = self.direction( vL, vR )

        mask = direction ==  1
        flux[ mask ] = qL[ mask ] * vL[ mask ]

        mask = direction == -1
        flux[ mask ] = qR[ mask ] * vR[ mask ]

        return flux


class Central2( Evolution1D ):
    def evolve( self, qL, qR, vL, vR ):
        flux = np.zeros_like( qL )

        flux[:]  = qL*vL + qR*vR
        flux[:] *= 0.5

        return flux


def LaxFriedrichs1( qL, qR, vL, vR, dx, dt ):
    """
    returns face flux evaluated using Lax-Friedrichs method on the cell face discontinuity
    """

    flux = qL*vL + qR*vR

    flux = flux - dx * ( qR - qL ) / dt

    flux = 0.5*flux

    return flux
