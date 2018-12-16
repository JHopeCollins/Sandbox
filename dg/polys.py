"""
Written by: Josh Hope-Collins ( joshua.hope-collins@eng.ox.ac.uk )

mish mash of things I probably need for dg code
to clean up later
"""

import numpy as np
import math
from scipy.special import gamma


def swap( a, b ):
    temp = a
    a = b
    b = temp
    return


class OrthonormalJacobi( object ):
    def GaussQuadrature( self, alpha, beta, N ):
        pass

    def GaussLobattoNodes( self, alpha, beta, N ):
        pass

    def a( self, alpha, beta, n ):
        """
        returns the coefficient a_n for the recurrence formula
        """

        c0 = 2.0 / ( 2.0*n + alpha + beta )

        c1 =      ( n + alpha + beta )
        c1 = c1 * ( n + alpha        )
        c1 = c1 * ( n         + beta )
        c1 = c1 *   n

        c2 = 2.0*n + alpha + beta - 1
        c3 = 2.0*n + alpha + beta + 1

        a = c0 * np.sqrt( c1 / ( c2*c3 ) )

        return a

    def b( self, alpha, beta, n ):
        """
        returns the coefficient b_n for the recurrence formula
        """

        c0 = alpha*alpha - beta*beta

        c1 = 2.0*n + alpha + beta
        c2 = 2.0*n + alpha + beta + 2

        b = c0 / ( c1*c2 )

        return -b

    def P0( self, alpha, beta, x ):
        """
        returns the value of the 0th order polynomial to start the recurrence
        """

        c0 = 2**( -( alpha + beta + 1 ) )

        c1 = gamma( alpha + beta + 2 )
        c2 = gamma( alpha        + 1 )
        c3 = gamma(         beta + 1 )

        p0 = np.sqrt( c0*c1 / ( c2*c3 ) )

        p0 = np.ones( len( x ) ) * p0

        return p0

    def P1( self, alpha, beta, x ):
        """
        returns the value of the 1st order polynomial to start the recurrence
        """

        c0 =        alpha + beta + 3
        c0 = c0 / ( alpha        + 1 )
        c0 = c0 / (         beta + 1 )
        c0 = np.sqrt( c0 )

        c1 = ( alpha + beta + 2 )*x
        c1 = c1 + ( alpha - beta )

        p0 = self.P0( alpha, beta, x )

        p1 = 0.5 * p0 * c0 * c1

        return p1

    def P( self, alpha, beta, n, x ):
        """
        return values of jacobi polynomial order n with parameters alpha, beta at absicca x.

        appendix A: discontinuous galerkin methods, algorithms, analysis and applications
        """
        if( n==0 ): return self.P0( alpha, beta, x )
        if( n==1 ): return self.P1( alpha, beta, x )

        p = np.zeros( [ n+1, len(x) ] )

        p[0,:] = self.P0( alpha, beta, x )
        p[1,:] = self.P1( alpha, beta, x )

        for i in range( 2, n+1 ):
            p[i,:] = p[i-1,:] *( x-self.b( alpha,beta,i-1 ))
            p[i,:] = p[i,  :] -    self.a( alpha,beta,i-1 )*p[i-2,:]
            p[i,:] = p[i,  :] /    self.a( alpha,beta,i   )

        return p[n,:]

    def dP( self, alpha, beta, n, x ):
        c0 = np.sqrt( n*( n + alpha + beta + 1 ) )
        return c0 * self.P( alpha+1, beta+1, n-1, x )

    def w( self, alpha, beta, x ):
        return (1-x)**alpha * (1+x)**beta


class OrthogonalJacobi( object ):
    def GaussQuadrature( self, alpha, beta, N ):
        pass

    def GaussLobattoNodes( self, alpha, beta, N ):
        pass

    def P0( self, alpha, beta, x ):
        return np.ones( len( x ) )

    def P1( self, alpha, beta, x ):
        c0 = 0.5*( alpha + beta + 2 )
        c1 = 0.5*( alpha - beta )
        return c0*x + c1

    def a( self, alpha, beta, n ):
        a  =   2.0*n + alpha + beta + 1.0
        a *= ( 2.0*n + alpha + beta + 2.0 )
        a /= (     n + alpha + beta + 1.0 )
        a /= (     n                + 1.0 )
        return 0.5*a

    def b( self, alpha, beta, n ):
        b  = beta*beta - alpha*alpha
        b *= ( 2.0*n + alpha + beta + 1.0 )
        b /= (     n + alpha + beta + 1.0 )
        b /= ( 2.0*n + alpha + beta       )
        b /= (     n                + 1.0 )
        return 0.5*b

    def c( self, alpha, beta, n ):
        c  =   2.0*n + alpha + beta + 2.0
        c *= (     n + alpha              )
        c *= (     n         + beta       )
        c /= ( 2.0*n + alpha + beta       )
        c /= (     n + alpha + beta + 1.0 )
        c /= (     n                + 1.0 )
        return c

    def P( self, alpha, beta, n, x ):
        """
        return the value of the nth order Jacobi polynomial with parameters alpha and beta at absicca x.

        values calculate using the recurrence relation given on page 74 of:
        http://lsec.cc.ac.cn/~hyu/teaching/shonm2013/STWchap3.2p.pdf
        """
        if ( n==0 ): return self.P0( alpha, beta, x )
        if ( n==1 ): return self.P1( alpha, beta, x )

        p = np.zeros( [n+1, len(x)] )

        p[0,:] = self.P0( alpha, beta, x )
        p[1,:] = self.P1( alpha, beta, x )

        for i in range( 2, n+1 ):
            a = self.a( alpha, beta, i-1 )
            b = self.b( alpha, beta, i-1 )
            c = self.c( alpha, beta, i-1 )

            p[i,:] = ( a*x - b )*p[i-1,:] - c*p[i-2,:]

        return p[n,:]

    def dP( self, alpha, beta, n, x ):
        return np.sqrt( n*(n+alpha+beta+1) )*self.P( 1, 1, n-1, x )

    def innerProduct( self, alpha, beta, i, j ):
        if( i != j ): return 0
        dot  =   2.0**(   alpha + beta + 1.0 )
        dot /= ( 2.0*i +  alpha + beta + 1.0 )

        dot *= gamma( i + alpha        + 1.0 )
        dot *= gamma( i +         beta + 1.0 )
        dot /= gamma( i + alpha + beta + 1.0 )
        
        dot /= math.factorial( i )

        return dot

    def w( self, alpha, beta, x ):
        return (1-x)**alpha * (1+x)**beta


class OrthogonalLegendre( object ):
    def GaussQuadrature( self, N ):
        pass

    def GaussLobattoNodes( self, N ):
        pass

    def P0( self, x ):
        return np.ones( len( x ) )

    def P1( self, x ):
        return x

    def P( self, x ):
        if( n==0 ): return self.P0( alpha, beta, x )
        if( n==1 ): return self.P1( alpha, beta, x )

        p = np.zeros( [ n+1, len(x) ] )

        p[0,:] = self.P0( alpha, beta, x )
        p[1,:] = self.P1( alpha, beta, x )

        for i in range( 2, n+1 ):
            p[i,:] = p[i-1,:] * x
            p[i,:] = p[i,  :] * ( 2*n - 1 )
            p[i,:] = p[i,  :] - (   n - 1 ) * p[i-2,:]
            p[i,:] = p[i,  :] / n

        return p[n,:]

    def dP( self, n, x ):
        return np.sqrt( n*(n+1) )*OrthogonalJacobi.P( 1, 1, n-1, x )

    def innerProduct( self, i, j ):
        if( i != j ): return 0

        return 2.0 / ( 2.0*i + 1 )


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


class Lagrange( object ):
    def interpolate( self, x, y ):
        self.x = x
        self.y = y
        self.order = len( x )
        self.d = np.zeros( self.order )
        self.f = np.zeros( [self.order, self.order] )

        for j in range( self.order ):
            self.d[j] = np.prod( np.delete( self.x[j] - self.x, j ) )
            self.d[j] = 1 / self.d[j]

            for i in np.delete( range( self.order ), j ):
                self.f[j, i] = np.prod( np.delete( self.x[j] - self.x, [i, j] ) )
                self.f[j, i] =     self.f[j, i] * ( x[j] - x[i] )
                self.f[j, i] = 1 / self.f[j, i]

        return

    def ljx( self, j, z ):
        p = np.prod( z - np.delete( self.x, j ) )

        return self.y[j] * self.d[j] * p

    def dljx( self, j, z):
        p0 = 0.0
        for i in np.delete( range( self.order ), j ):
            p1  = np.prod( np.delete( z - self.x, [i, j] ) )
            p1 *= self.f[j, i]
            p0 += p1

        return self.y[j] * p0

    def yi( self, z ):
        p = map( lambda j : self.ljx( j, z ), range( self.order ) )
        return sum( p )

    def dyi( self, z):
        p = map( lambda j : self.dljx( j, z ), range( self.order ) )
        return sum( p )


def lagrange_polynomial( x, y, j ):
    """
    return a function which is the jth lagrange interpolating polynomial for data y at absicca x
    """

    d = 1 / np.prod( np.delete( x[j] - x, j ) )

    poly = lambda z : y[j] * d * np.prod( z - np.delete( x, j ) )

    # poly = lambda z : y[j] * np.prod( np.repeat(                 z[np.newaxis, :], len( x ) -1, axis=0 ) -
    #                                   np.repeat( np.delete( x, j )[:, np.newaxis], len( z ),    axis=1 ), axis=0 ) * d

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

def lagrange_derivative( x, y, j ):
    """
    returns a function which is the derivative of the jth lagrange interpolating polynomial for data y at absicca x
    """

    xwij = []
    for i in np.delete( range( len( x ) ), j ):
        xwij.append( np.delete( x, [ i, j ] ) )

    d = map( lambda xk : np.prod( x[j] - xk ), xwij )

    d = 1 / ( ( x[j] - np.delete( x, j ) ) * d )

    der = lambda z : y[j] * sum( d * map( lambda xk : np.prod(z - xk), xwij ) )

    return der

def lagrange_derivatives( x, y ):
    """
    returns a list of functions which are the derivatives of the lagrange interpolating polynomials for data y at absicca x
    """

    ders = []
    for j in range( len ( x ) ):
        ders.append( lagrange_derivative( x, y, j ) )

    return ders

def lagrange_interp_deriv( xi, yi ):
    """
    returnts a function which is the derivative of the lagrange interpolating polynomial of data y at absicca x
    """

    ders = lagrange_derivatives( xi, yi )

    deriv = lambda x : sum( map( lambda d : d(x), ders ) )

    return deriv

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
    return lambda X : 0.5*(X+1.0)*dx + xL

def G1D( xL, xR ):
    """
    return a lambda function which maps from x E [xL,xR] -> X E [-1,1]
    """
    dx = xR - xL
    return lambda x : 2.0*(x-xL)/dx - 1.0

