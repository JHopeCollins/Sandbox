"""
Written by: Josh Hope-Collins ( joshua.hope-collins@eng.ox.ac.uk )

simulation of sod's shock tube problem
"""

import numpy as np
import matplotlib.pyplot as plt

import sandbox as sb

def blank( x ):
    fig1, ax1 = plt.subplots(1,1)

    for i in range( len( x.xh ) ):
        xx = [ x.xh[i], x.xh[i] ]
        yy = [ -0.1, 2.6 ]
        ax1.plot( xx, yy, color='lightgrey' )

    return fig1, ax1

# domain and boundary conditions
nx  = 30
cfl = 0.05
nt  = 10

lb =   nx / 4
ub = 3*nx / 4

x = np.linspace( -1, 1, nx+1 )
x = sb.fv.fields.Domain( x )

bc0 = sb.general.fields.BoundaryCondition( name='naive_equal', indx= 0 )
bc1 = sb.general.fields.BoundaryCondition( name='naive_equal', indx=-1 )

# conserved variables
#   density
#   momentum
#   energy

q = sb.fv.euler.EulerField1D( x )
q.add_boundary_condition( bc0 )
q.add_boundary_condition( bc1 )

# initial conditions
#   density
#   velocity
#   pressure
rhoL, rhoR = 1.0, 0.125
uL,   uR   = 0.0, 0.0
pL,   pR   = 1.0, 0.1

TL = pL / ( q.R * rhoL )
TR = pR / ( q.R * rhoR )
eL = q.Cv*TL
eR = q.Cv*TR
EL = eL + 0.5*uL*uL
ER = eR + 0.5*uR*uR

assert nx % 2 == 0
ones = np.ones( nx/2 )
data = np.zeros_like( q.val )
data[0,:] = np.append( ones*rhoL,    ones*rhoR    )
data[1,:] = np.append( ones*rhoL*uL, ones*rhoR*uR )
data[2,:] = np.append( ones*rhoL*EL, ones*rhoR*ER )

q.set_field( data )

fig1, ax1 = blank( x )
fig2, ax2 = blank( x )
fig3, ax3 = blank( x )

ax1.plot( x.xp[:], q[0][:], label=r'$\rho_0$'   )
ax1.plot( x.xp[:], q[1][:], label=r'$\rho U_0$' )
ax1.plot( x.xp[:], q[2][:], label=r'$\rho E_0$' )
ax1.plot( x.xp[:], q.p[:],  label=r'$p_0$'      )

ax2.plot( x.xp[:], q[0][:], label=r'$\rho_0$'   )
ax2.plot( x.xp[:], q[1][:], label=r'$\rho U_0$' )
ax2.plot( x.xp[:], q[2][:], label=r'$\rho E_0$' )
ax2.plot( x.xp[:], q.p[:],  label=r'$p_0$'      )

ax3.plot( x.xp[:], q[0][:], label=r'$\rho_0$'   )
ax3.plot( x.xp[:], q[1][:], label=r'$\rho U_0$' )
ax3.plot( x.xp[:], q[2][:], label=r'$\rho E_0$' )
ax3.plot( x.xp[:], q.p[:],  label=r'$p_0$'      )

a  = max( q.a[:] )
dx = min( x.dxh )
dt = cfl*dx / a

# upwind flux
flux = sb.fv.euler.EulerUDS1( q )
step = sb.fv.ODEintegrators.EulerForward1

def L( q ):
    shape = q.val.shape * np.ones_like( q.val.shape ) 
    shape[-1] += 1
    f = np.zeros( shape )

    f[:] += flux.apply( q )

    return np.diff( f ) / q.mesh.dxh

f = flux.apply( q )
ax1.plot( x.xh[:], f[0,:], label=r'$f_{\rho}$'   )
ax1.plot( x.xh[:], f[1,:], label=r'$f_{\rho U}$' )
ax1.plot( x.xh[:], f[2,:], label=r'$f_{\rho E}$' )

d = step( dt, q, L )
ax2.plot( x.xp[:], d[0,:], label=r'$\Delta \rho$'   )
ax2.plot( x.xp[:], d[1,:], label=r'$\Delta \rho U$' )
ax2.plot( x.xp[:], d[2,:], label=r'$\Delta \rho E$' )

    
for n in range( nt ):
    dq = step( dt, q, L )
    q.update( dq )

ax3.plot( x.xp[:], q[0][:], label=r'$\rho_1$'   )
ax3.plot( x.xp[:], q[1][:], label=r'$\rho U_1$' )
ax3.plot( x.xp[:], q[2][:], label=r'$\rho E_1$' )
ax3.plot( x.xp[:], q.p[:],  label=r'$p_1$'      )

ax1.legend()
ax1.set_title( 'flux' )
fig1.show()

ax2.legend()
ax2.set_title( 'difference' )
fig2.show()

ax3.legend()
ax3.set_title( 'change' )
fig3.show()
