"""
Written by: J Hope-Collins ( joshua.hope-collins@eng.ox.ac.uk )

procedural script for first try at dg for linear advection wave
"""
import numpy as np
import matplotlib.pyplot as plt
from dg import polys
from fv import evolutions
from fv import ODEintegrators

def RK4( dt, q, L ):
    dq1 = ODEintegrators.EulerForward1( dt, q, L )
    r = q.copy() + 0.5*dq1

    dq2 = ODEintegrators.EulerForward1( dt, r, L )
    r = q.copy() + 0.5*dq2

    dq3 = ODEintegrators.EulerForward1( dt, r, L )
    r = q.copy() + dq3

    dq4 = ODEintegrators.EulerForward1( dt, r, L )

    return (dq1 + 2*dq2 + 2*dq3 + dq4)*0.166666666667

# x is physical space
# z is computational space for each cell [-1, 1]
# h refers to cell data
# p refers to nodal data

nh  = 20        # number of cells
p   = 3         # order of approximation
L   = 2*np.pi   # domain length
T   = 0.2       # time duration (# of periods)
c   = 1         # advection velocity
cfl = 0.05      # CFL number

dx = L / nh
dt = cfl * dx / c
T *= L/c
nt = int( T/dt )

# numerical face flux and time stepping routines
flux = evolutions.upwind1
# step = ODEintegrators.EulerForward1
step = RK4

#cell vertices
xh = np.linspace(0, L, nh+1 )

#nodal points and weights
xp = np.zeros( nh*p )
zp = polys.Legendre.zeros[   p ]
wp = polys.Legendre.weights[ p ]

M = np.zeros(  nh*p  )    # mass      matrix
S = np.zeros( [p, p] )    # stiffness matrix

lag = polys.Lagrange()
lag.interpolate( zp, np.ones( p ) )

# extrapolated lagrange basis boundary values for flux calculation
Lm1 = np.zeros( nh * p )
Lp1 = np.zeros( nh * p )
tLm1 = map( lambda j : lag.ljx( j, -1 ), range( p ) )
tLp1 = map( lambda j : lag.ljx( j,  1 ), range( p ) )

# construct global mass matrix, nodal points and extrapolation vectors
for k in range( nh ):
    kL = p*k
    kR = kL + p

    g = polys.g1D( xh[k], xh[k+1] )

    xp[ kL : kR ] = g( zp )

    dg = xh[k+1] - xh[k]
    dg *= 0.5

    M[ kL : kR ] = dg*wp

    Lm1[ kL : kR ] = tLm1
    Lp1[ kL : kR ] = tLp1

# initialise nodal values
qp = np.sin( xp )

# face flux from extrapolated values
def faceflux( q, c ):
    um = Lm1.reshape( nh, p ) * q.reshape( nh, p )
    up = Lp1.reshape( nh, p ) * q.reshape( nh, p )

    um = np.sum( um, axis=1 )
    up = np.sum( up, axis=1 )

    uL = np.append( up[-1], up    )
    uR = np.append( um,     um[0] )

    ct = c*np.ones( nh+1 )

    f   = flux( uL, uR, ct, ct )

    f   = np.repeat( f, p )
    fn  = f[p: ] * Lp1
    fn -= f[:-p] * Lm1

    return fn

# construct cell stiffness matrix
temp = np.zeros( [p, p] )
for m in range( p ):
    temp[:,:] = 0.0
    for i in range( p ):
        for j in range( p ):
            if i==m:
                temp[ j, i ] = lag.dljx( j, zp[m] )*wp[m]
    S +=temp

# construct global stiffness matrix
S = np.kron( np.eye( nh ), S )

# spatial operator
def L( q ):
    fn = faceflux( q, c )
    return -( c * np.matmul( S, q ) - fn ) / M

for n in range( nt ):
    dq = step( dt, qp, L )
    qp += dq

plt.plot(xp,qp)
plt.show()
