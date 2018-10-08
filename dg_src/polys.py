
import numpy as np

class Legendre( object ):
    """
    class for Legendre polynomials, including zeros and weights for Legendre-Gauss quadrature
    """

    zeros = [[ 0 ],
             [ 0 ],
             [ -0.577350269189626, 0.577350269189626 ],
             [ -0.774596669241483, 0, 0.774596669241483 ],
             [ -0.86113631159405, -0.339981043584856, 0.339981043584856, 0.86113631159405 ],
             [ -0.90617984593866, -0.538469310105683, 0, 0.538469310105683, 0.90617984593866 ],
             [ -0.932469514203152, -0.661209386466265, -0.238619186083197, 0.238619186083197, 0.661209386466265, 0.932469514203152 ]]

    zeros = np.asarray( zeros )

    weights = [[ 1 ],
               [ 1 ],
               [ 1, 1 ],
               [ 0.555555555555556, 0.888888888888889, 0.555555555555556 ],
               [ 0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454 ],
               [ 0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189 ],
               [ 0.171324492379170, 0.360761573048139, 0.467913934572691, 0.467913934572691, 0.360761573048139, 0.171324492379170 ]]

    weights = np.asarray( weights )

    def LG_quadrature( self, n, k ):
        """
        return the kth root and Gauss quadrature weight of the nth order Legendre polynomial

        Values obtained from:
        Lowan, Davids, Levenson, American Mathematical Society 1941, 'Table of the zeros of the legendre polynomials of order 1-1 and the weight coefficients for Gauss' mechanical quadrature formula'
        """
        return self.zeros[ n ][ k ], self.weights[ n ][ k ]

def lagrange_polynomial( x, y, j ):
    """
    return a function which is the jth lagrange interpolating polynomial for data y at absicca x
    """

    d = np.prod( np.delete( x[j] - x, j ) )

    # old line for float z, replaced by line below for array z
    # poly = lambda z : y[j] * np.prod( z - np.delete( x, j ) ) / d

    poly = lambda z : y[j] * np.prod( np.repeat( z[np.newaxis, :],                 len( x ) -1, axis=0 ) -
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
