"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Reconstruction functions for REA type finite volume numerical schemes
Each function takes in a single array of cell average values and returns left and right face values for each cell
"""

import numpy as np
import maths_utils as mth

def PCM1( u, dx, h ):
    """
    returns left and right cell face values for piecewise constant reconstruction

    reconstruction_radius = 1
    """
    uR = u[1:]
    uL = u[:-1]
    return uL, uR

def minmod2( u, dx, h ):
    """
        returns left and right cell face values for piecewise linear reconstruction with minmod choice of slopes

        reconstruction_radius = 2

        minmod chooses slope of linear reconstruction to be minimum of neighbouring slopes, or zero if cell at minimum/maximum (when slopes of opposite sign)
    """

    sigmaL = ( u[1:-1] - u[ :-2] ) / dx[ :-1]
    sigmaR = ( u[2:  ] - u[1:-1] ) / dx[1:  ]

    sigma  = mth.minmod( sigmaL, sigmaR )

    du = 0.5*sigma*h[1:-1]
    uR = u[ 2:-1 ] - du[ :-1]
    uL = u[ 1:-2 ] + du[1:  ]

    return uL, uR

def superbee2( u, dx, h ):
    """
        returns left and right cell face values for piecewise linear reconstruction with superbee choice of slopes

        reconstruction_radius = 2

        superbee chooses slope of linear reconstruction to be minimum of each one-sided slope compared with twice the other one-sided slope, or zero if cell at minimum/maximum (when slopes of opposite sign)
        This choice of slopes tends to choose the larger slope in smooth regions, but still the smaller slope near discontinuities
    """

    sigmaL = ( u[1:-1] - u[ :-2] ) / dx[ :-1]
    sigmaR = ( u[2:  ] - u[1:-1] ) / dx[1:  ]

    sigma1  = mth.minmod(   sigmaL, 2*sigmaR )
    sigma2  = mth.minmod( 2*sigmaL,   sigmaR )

    sigma = mth.maxmod( sigma1, sigma2 )

    du = 0.5*sigma*h[1:-1]
    uR = u[ 2:-1 ] - du[ :-1]
    uL = u[ 1:-2 ] + du[1:  ]

    return uL, uR

def MC2( u, dx, h ):
    """
        returns left and right cell face values for piecewise linear reconstruction with monotized central-difference limiter choice of slopes

        reconstruction_radius = 2

        MC chooses slope of linear reconstruction to be minimum of twice each one-sided slope or the central difference slope, or zero if cell at minimum/maximum (when slopes of opposite sign)
        This choice of slopes tends to choose the central slope in smooth regions, which results in better resolution than the superbee choice, which tends to artificially steepen smooth regions
    """

    sigmaL = ( u[1:-1] - u[ :-2] ) / dx[1:  ]
    sigmaR = ( u[2:  ] - u[1:-1] ) / dx[ :-1]
    sigmaC = ( u[2:  ] - u[ :-2] ) / ( dx[1:] + dx[:-1] )

    sigma = mth.minmod3( 2*sigmaL, 2*sigmaR, sigmaC )

    du = 0.5*sigma*h[1:-1]
    uR = u[ 2:-1 ] - du[ :-1]
    uL = u[ 1:-2 ] + du[1:  ]

    return uL, uR

