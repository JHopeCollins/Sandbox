
import numpy as np
from dg import polys
from fv import evolutions
from fv import ODEintegrators

# create mesh

# x is physical space
# z is computational space for each cell [-1, 1]
# h refers to cell data
# p refers to nodal data

nh = 20
p  = 5

#cell vertices
xh = np.linspace(0, 2*np.pi, nh+1 )

#nodal points
xp = np.zeros( nh*p )
zp = polys.Legendre.zeros[ p ]

for i in range( nh ):
    g = polys.g1D( xh[i], xh[i+1] )

    xp[ i*p : (i+1)*p ] = g( zp )


# calculate cell mass matrices - GL weights and cell jacobian

# calculate cell stiffness matrices - Lagrange interpolations derivatives

# calculate extrapolation matrices for cell edge values

# get flux function from /fv/evolutions.py

# initialise nodal values from sinusoid

# create function dqdt for time-derivative of nodal values
#   L(q) = (c*K*q - f)/(w*g')

# get ODEintegrator from /fv/ODEintegrators.py

# for n in nt:
#   q =+ ODEintegrator( L(q) )

