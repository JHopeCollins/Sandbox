"""
Written by: J Hope-Collins (joshua.hope-collins@eng.ox.ac.uk)

Test suite for polys.py
"""

import numpy as np
import polys

class Test_Legendre( object ):
    def test_weights( self ):
        L = polys.Legendre()

        assert type( L.weights ) == np.ndarray
        assert L.weights[ 0, 0 ] == 1
        assert L.weights[ 4, 1 ] == 0.888888888888889
        assert L.weights[ 6, 3 ] == 0.467913934572691

        return

    def test_zeros( self ):
        L = polys.Legendre()

        assert type( L.zeros ) == np.ndarray
        assert L.zeros[ 1, 0 ] == 0
        assert L.zeros[ 3, 0 ] == -0.774596669241483
        assert L.zeros[ 5, 2 ] == 0

        return

