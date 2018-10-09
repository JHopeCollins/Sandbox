import numpy as np
import matplotlib.pyplot as plt
import polys


def runge( x ): return 1.0/(1.0 + 25.0*x*x)

x = np.linspace( -1, 1, 101 )
y = runge( x )


xi0  = np.linspace( -1, 1, 11 )
lag0 = polys.lagrange_interpolator( xi0, runge( xi0 ) )
yi0  = lag0( x )

xi1  = np.asarray( [ -1.0, -0.95, -0.81, -0.59, -0.31, 0.0, 0.31, 0.59, 0.81, 0.95, 1.0 ] )
lag1 = polys.lagrange_interpolator( xi1, runge( xi1 ) )
yi1  = lag1( x )

