"""
Written by J Hope-Collins (jth39@cam.ac.uk)

Test suite for jump_fluxes.py
"""

import numpy as np
import jump_fluxes as jf

def test_upwind1( ):
    qL = np.asarray( [ 1,  2,  3,  4] )
    qR = np.asarray( [11, 12, 13, 14] )

    #                  L   R   R   L
    vL = np.asarray( [ 1,  1,  1,  1] )
    vR = np.asarray( [ 2, -2, -2,  2] )

    h  = np.asarray( [ 1,  1,  1,  1, 1] )
    dx = 1

    f = jf.upwind1( qL, qR, vL, vR, dx, h )

    assert np.all( f == [ 1, -24, -26, 4 ] )

    return

def test_LaxFriedrichs1( ):
    qL = np.asarray( [ 1,  2,  3] )
    qR = np.asarray( [11, 12, 13] )

    vL = np.asarray( [ 1,  1,  1] )
    vR = np.asarray( [ 2, -2, -2] )

    h  = 1
    dx = 0.5
    dt = 0.1

    f = jf.LaxFriedrichs1( qL, qR, vL, vR, dx, dt )

    ftest= []
    ftest.append( ( 1 + 22 - 0.5*10/0.1 )*0.5 )
    ftest.append( ( 2 - 24 - 0.5*10/0.1 )*0.5 )
    ftest.append( ( 3 - 26 - 0.5*10/0.1 )*0.5 )

    assert np.all( f == ftest )

    return

