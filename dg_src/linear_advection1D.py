
import numpy as np
import polys

# create mesh

# x is physical space
# z is computational space for each cell [-1, 1]
# h refers to cell data
# p refers to nodal data

nh = 20
p  = 5

xh = np.linspace(0, 2*np.pi, nh+1 )
xp = np.zeros( nh*p )

zp = polys.Legendre.zeros[ p ]

for i in range( nh ):
    il = i*p
    ir = il+p

    G = polys.G1D( xh[i], xh[i+1] )

    xp[il:ir] = G( zp )

# calculate cell jacobians

# calculate mass matrix - GL weights

# calculate stiffness matrix - need function for derivative of Lagrange interpolations

# get flux function from /fv_src/evolutions.py

# initialise nodal values from sinusoid

# create function dqdt for time-derivative of nodal values
#   L(q) = (c*K*q - f)/(w*g')

# get ODEintegrator from /fv_src/ODEintegrators.py

# for n in nt:
#   q =+ ODEintegrator( L(q) )

