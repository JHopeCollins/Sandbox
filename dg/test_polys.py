"""
Written by: J Hope-Collins (joshua.hope-collins@eng.ox.ac.uk)

Test suite for polys.py
"""

import numpy as np
import polys

def runge( x ): return 1 / ( 1 + 25*x*x )

eps = 0.00000000000001

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

    def test_GL_quad_0w( self ):
        L = polys.Legendre()

        assert L.GL_quad_0w(0,0) == ( L.zeros[0][0], L.weights[0][0] )

        assert L.GL_quad_0w(4,3) == ( L.zeros[4][3], L.weights[4][3] )

class Test_Lagrange( object ):
    def test_interpolate( self ):
        x = np.linspace( -1, 1, 11 )
        y = runge( x )

        L = polys.Lagrange()
        L.interpolate( x, y )

        assert np.all( L.x == x )
        assert np.all( L.y == y )

        assert len( L.d == len( x ) )

        assert L.d[2] == 1 / np.prod( np.delete( x[2] - x, 2 ) )

        return

    def test_ljx( self ):
        x = np.linspace( -1, 1, 11 )
        y = runge( x )

        L = polys.Lagrange()
        L.interpolate( x, y )

        for j in range( len ( x ) ):

            yi = map( lambda z : L.ljx( j, z ), x )

            assert np.isclose( yi[j], y[j] )

            assert np.all( np.delete( yi, j ) == 0 )

        return

    def test_dljx( self ):
        x = np.linspace( -1, 1, 11 )
        y = runge( x )

        L = polys.Lagrange()
        L.interpolate( x, y )

        for j in range( len( x ) ):

            dy  = np.asarray( map( lambda z : L.dljx( j, z ), x ) )

            dyt = np.asarray( map( lambda z : ( L.ljx(j,z+eps) - L.ljx(j,z-eps) )/( 2*eps ), x ) )

            err = np.abs( [ (dy[k]-dyt[k])/dyt[k] for k in range(len(x)) if dy[k]>eps*1000 ] )
            assert np.all( err  < 0.025 )

        return

    def test_yi( self ):
        x = np.linspace( -1, 1, 11 )
        y = runge( x )

        L = polys.Lagrange()
        L.interpolate( x, y )

        yi = map( lambda z : L.yi( z ), x )

        assert np.all( np.isclose( yi, y ) )

        assert L.yi( 0.7 ) + 0.226 < 0.001

        return

    def test_dyi( self ):
        x = np.linspace( -1, 1, 11 )
        y = runge( x )

        L = polys.Lagrange()
        L.interpolate( x, y )

        for j in range( len( x ) ):

            d = L.dyi( x[j] )

            dt = sum( map( lambda k : L.dljx( k, x[j] ), range( len( x ) ) ) )

            assert d == dt

        return

def test_lagrange_polynomial( ):
    x = np.linspace( -1, 1, 11 )

    for i in range( len( x ) ):

        l = polys.lagrange_polynomial( x, runge( x ), i )

        yi = map( lambda xj : l( xj ),  x )

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

        yc = map( lambda xj :   li(xj), x_test )
        yt = map( lambda xj : l[i](xj), x_test )

        assert np.all( yc == yt )

    return

def test_lagrange_interpolator( ):
    x = np.linspace( -1, 1, 11 )

    interp = polys.lagrange_interpolator( x, runge( x ) )

    yi = map( lambda xj : interp(xj), x )

    assert np.all( np.isclose( yi, runge( x ) ) )

    assert interp( 0.7 ) + 0.226 < 0.001

    return

def test_lagrange_derivative( ):
    x = np.linspace( -1, 1, 11 )

    for i in range( len( x ) ):
        der = polys.lagrange_derivative( x, runge( x ), i )
        l   = polys.lagrange_polynomial( x, runge( x ), i )

        dy  = np.asarray( map( lambda xj : der(xj), x ) )
        dyt = np.asarray( map( lambda xj : ( l(xj+eps) - l(xj-eps) )/( 2*eps ), x ) )

        err = np.abs( [ (dy[j]-dyt[j])/dyt[j] for j in range(len(x)) if dy[j]>eps*1000 ] )
        assert np.all( err  < 0.025 )

    return

def test_lagrange_derivatives( ):
    x = np.linspace( -1, 1, 11 )

    x_test = np.linspace( -1, 1, 14 )

    d = polys.lagrange_derivatives( x, runge( x ) )

    assert len( d ) == len( x )

    for i in range( len( x ) ):

        di = polys.lagrange_derivative( x, runge( x ), i )

        dc = map( lambda xj :   di(xj), x_test )
        dt = map( lambda xj : d[i](xj), x_test )

        assert np.all( dc == dt )

    return

def test_lagrange_interp_deriv( ):
    x = np.linspace( -1, 1, 11 )

    d = polys.lagrange_interp_deriv( x, runge( x ) )
    ds = polys.lagrange_derivatives( x, runge( x ) )

    di = map( lambda xj : d(xj), x )

    for i, xi in enumerate( x ):
        dt = sum( map( lambda p : p(xi), ds ) )
        assert dt == di[i]

    return

def test_GLquad( ):
    i = polys.GLquad( lambda x : np.log(x)/x, 5, [1,8] )
    assert i - 2.165234 < 0.000001
    return

def test_g1D( ):
    g = polys.g1D( -2, 2 )

    assert np.isclose( g( -0.5 ), -1  )
    assert np.isclose( g(  0.3 ), 0.6 )

    g = polys.g1D( 0, 2 )

    assert np.isclose( g( -0.1 ), 0.9 )
    assert np.isclose( g(  0.8 ), 1.8 )

    g = polys.g1D( 0, 4 )

    assert np.isclose( g( -0.4 ), 1.2 )
    assert np.isclose( g(  0.6 ), 3.2 )

    return

def test_G1D( ):
    G = polys.G1D( -2, 2 )

    assert np.isclose( G( -1  ), -0.5 )
    assert np.isclose( G( 0.6 ),  0.3 )

    G = polys.G1D( 0, 2 )

    assert np.isclose( G( 0.9 ), -0.1 )
    assert np.isclose( G( 1.8 ),  0.8 )

    G = polys.G1D( 0, 4 )

    assert np.isclose( G( 1.2 ), -0.4 )
    assert np.isclose( G( 3.2 ),  0.6 )



