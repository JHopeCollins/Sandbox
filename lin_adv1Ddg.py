"""
Written by: J Hope-Collins ( joshua.hope-collins@eng.ox.ac.uk )

script for dg linear advection wave using dg classes
"""

import numpy as np
import matplotlib.pyplot as plt
from dg import polys
from dg import fields
from fv import evolutions
from fv import ODEintegrators

nh = 5
p  = 6

T  = 10
L  = 2*np.pi

c   = 1
cfl = 0.05

dx = L / nh
dt = cfl * dx / c
nt = int( T/dt )

flux = evolutions.upwind1
step = ODEintegrators.RungeKutta4

mesh = np.linspace( 0, L, nh+1 )
mesh = fields.Domain( mesh )
mesh.set_expansion( p )

q = fields.UnsteadyField1D( 'q', mesh )
q.set_field( np.sin( mesh.xp ) )
q.set_timestep( dt )
q.set_save_interval( dt=0.05/(2*np.pi) )

# face flux from extrapolated values
def faceflux( q, c ):
    um = mesh.Lm1.reshape( nh, p ) * q.val.reshape( nh, p )
    up = mesh.Lp1.reshape( nh, p ) * q.val.reshape( nh, p )

    um = np.sum( um, axis=1 )
    up = np.sum( up, axis=1 )

    uL = np.append( up[-1], up    )
    uR = np.append( um,     um[0] )

    ct = c*np.ones( nh+1 )

    f   = flux( uL, uR, ct, ct )

    f   = np.repeat( f, mesh.p )
    fn  = f[p: ] * mesh.Lp1
    fn -= f[:-p] * mesh.Lm1

    return fn

lag = polys.Lagrange()
lag.interpolate( mesh.zp, np.ones( mesh.p ) )

# construct cell stiffness matrix
S    = np.zeros( [p, p] )
temp = np.zeros( [p, p] )
for m in range( p ):
    temp[:,:] = 0.0
    for i in range( p ):
        for j in range( p ):
            if i==m:
                temp[ j, i ] = lag.dljx( j, mesh.zp[m] )*mesh.wp[m]
    S +=temp

# construct global stiffness matrix
S = np.kron( np.eye( nh ), S )

# spatial operator
def L( q ):
    fn = faceflux( q, c )
    return -( c * np.matmul( S, q.val ) - fn ) / mesh.M

for n in range( nt ):
    dq = step( dt, q, L )
    q.update( dq )

q.plot_history()


