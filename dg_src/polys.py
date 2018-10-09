"""
Written by: Josh Hope-Collins ( joshua.hope-collins@eng.ox.ac.uk )
"""

import numpy as np

class Legendre( object ):
    """
    class for Legendre polynomials, including zeros and weights for Legendre-Gauss quadrature
    """

    zeros = np.asarray( [ np.asarray( [ 0 ] ),
            np.asarray( [ 0 ] ),
            np.asarray( [ -0.577350269189626, 0.577350269189626 ] ),
            np.asarray( [ -0.774596669241483, 0, 0.774596669241483 ] ),
            np.asarray( [ -0.861136311594053, -0.339981043584856,    0.339981043584856, 0.861136311594053 ] ),
            np.asarray( [ -0.906179845938664, -0.538469310105683, 0, 0.538469310105683, 0.906179845938664 ] ),
            np.asarray( [ -0.932469514203152, -0.661209386466265, -0.238619186083197,    0.238619186083197, 0.661209386466265, 0.932469514203152 ] ),
            np.asarray( [ -0.949107912342759, -0.741531185599394, -0.405845151377397, 0, 0.405845151377397, 0.741531185599394, 0.949107912342759 ] ),
            np.asarray( [ -0.960289856497536, -0.796666477413627, -0.525532409916329, -0.183434642495650, 0.183434642495650, 0.525532409916329, 0.796666477413627, 0.960289856497536 ] ) ] )


    weights = np.asarray( [ np.asarray( [ 1 ] ),
              np.asarray( [ 1 ] ),
              np.asarray( [ 1, 1 ] ),
              np.asarray( [ 0.555555555555556, 0.888888888888889, 0.555555555555556 ] ),
              np.asarray( [ 0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454 ] ),
              np.asarray( [ 0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189 ] ),
              np.asarray( [ 0.171324492379170, 0.360761573048139, 0.467913934572691, 0.467913934572691, 0.360761573048139, 0.171324492379170 ] ),
              np.asarray( [ 0.129484966168870, 0.279705391489277, 0.381830050505119, 0.417959183673469, 0.381830050505119, 0.279705391489277, 0.129484966168870 ] ),
              np.asarray( [ 0.101228536290376, 0.222381034453374, 0.313706645877887, 0.362683783378362, 0.362683783378362, 0.313706645877887, 0.222381034453374, 0.101228536290376 ] ) ] )


    def GL_quad_0w( self, n, k ):
        """
        return the kth root and Gauss quadrature weight of the nth order Legendre polynomial

        Values obtained from:
        Lowan, Davids, Levenson, American Mathematical Society 1941, 'Table of the zeros of the legendre polynomials of order 1-16 and the weight coefficients for Gauss' mechanical quadrature formula'
        """
        return self.zeros[ n ][ k ], self.weights[ n ][ k ]

def lagrange_polynomial( x, y, j ):
    """
    return a function which is the jth lagrange interpolating polynomial for data y at absicca x
    """

    d = np.prod( np.delete( x[j] - x, j ) )

    # old line for float z, replaced by line below for array z
    # poly = lambda z : y[j] * np.prod( z - np.delete( x, j ) ) / d

    poly = lambda z : y[j] * np.prod( np.repeat(                 z[np.newaxis, :], len( x ) -1, axis=0 ) -
                                      np.repeat( np.delete( x, j )[:, np.newaxis], len( z ),    axis=1 ), axis=0 ) / d

    return poly

def lagrange_polynomials( x, y ):
    """
    return a list of functions which are the lagrange interpolating polynomials for data y at absicca x
    """

    polys = []
    for j in range( len( x ) ):
        polys.append( lagrange_polynomial( x, y, j ) )

    return polys

def lagrange_interpolator( xi, yi ):
    """
    returns a function which is the lagrange polynomial interpolation for data y at absicca x
    """
    polys = lagrange_polynomials( xi, yi)

    interpolator = lambda x : sum( map( lambda p : p(x), polys ) )

    return interpolator

def GLquad( f, n, bounds=[-1, 1] ):
    """
    returns the Gauss Legendre quadrature with n points for function f(x) and bounds[lower, upper]
    """

    a = bounds[0]
    b = bounds[1]

    z = Legendre.zeros[   n ]
    w = Legendre.weights[ n ]

    x = 0.5*( (b+a) + z*(b-a) )

    return 0.5*(b-a)*sum( f(x)*w )

def g1D( xL, xR ):
    """
    return a lambda function which maps from X E [-1,1] -> x E [xL,xR]
    """
    dx = xR - xL
    return lambda X : 0.5*(X+1)*dx + xL

def G1D( xL, xR ):
    """
    return a lambda function which maps from x E [xL,xR] -> X E [-1,1]
    """
    dx = xR - xL
    return lambda x : 1 + 2*(x - xL)/dx

