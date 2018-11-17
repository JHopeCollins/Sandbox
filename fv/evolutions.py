"""
Written by J Hope-Collins (jth39@cam.ac.uk)

Numerical flux functions for REA type finite volume schemes
Each function takes arrays of left and right values of advection velocity and advected quantity at each face, and returns the numerical flux associated with this jump
"""

import numpy as np

def upwind1( qL, qR, vL, vR, ):
    """
    returns face flux calculated from upwind value on face
    """

    flux = np.zeros_like( qL )

    direction = np.sign( vL + vR ).astype( int )

    mask = direction ==  1
    flux[ mask ] = qL[ mask ] * vL[ mask ]

    mask = direction == -1
    flux[ mask ] = qR[ mask ] * vR[ mask ]

    mask = direction == 0
    flux[ mask ] = qR[ mask ] * vR[ mask ]

    return flux

def LaxFriedrichs1( qL, qR, vL, vR, dx, dt ):
    """
    returns face flux evaluated using Lax-Friedrichs method on the cell face discontinuity
    """

    flux = qL*vL + qR*vR

    flux = flux - dx * ( qR - qL ) / dt

    flux = 0.5*flux

    return flux
