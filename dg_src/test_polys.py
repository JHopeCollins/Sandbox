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
        assert L.weights[ 0 ][ 0 ] == 1
        assert L.weights[ 4 ][ 1 ] == 0.652145154862546
        assert L.weights[ 6 ][ 3 ] == 0.467913934572691

        return

    def test_zeros( self ):
        L = polys.Legendre()

        assert type( L.zeros ) == np.ndarray
        assert L.zeros[ 1 ][ 0 ] == 0
        assert L.zeros[ 3 ][ 0 ] == -0.774596669241483
        assert L.zeros[ 5 ][ 2 ] == 0

        return

    def test_LG_quadrature( self ):
        L = polys.Legendre()

        assert L.LG_quadrature(0,0) == ( L.zeros[0][0], L.weights[0][0] )

        assert L.LG_quadrature(4,3) == ( L.zeros[4][3], L.weights[4][3] )


def runge( x ): return 1 / ( 1 + 25*x*x )

def test_lagrange_polynomial( ):
    x = np.linspace( -1, 1, 11 )

    for i in range( len( x ) ):
        l = polys.lagrange_polynomial( x, runge( x ), i )
        yi = l( x )

        assert np.isclose( yi[i], runge( x[i] ) )

        assert np.all( np.delete( yi, i ) == 0 )

    return

def test_lagrange_polynomials( ):
    x = np.linspace( -1, 1, 11 )

    x_test = np.linspace( -1, 1, 14 )

    l = polys.lagrange_polynomials( x, runge( x ) )

    assert len( l ) == len( x )

    for i in range( len( x ) ):

        li = polys.lagrange_polynomial( x, runge( x ), i )

        assert np.all( l[i]( x_test ) == li( x_test ) )

    return

def test_lagrange_interpolator( ):
    x = np.linspace( -1, 1, 11 )

    interp = polys.lagrange_interpolator( x, runge( x ) )

    assert np.all( np.isclose( interp( x ), runge( x ) ) )

    assert interp( np.asarray([0.7]) ) + 0.226 < 0.001

    return

