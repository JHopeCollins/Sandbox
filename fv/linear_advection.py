"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Finite volume approximation to 1D scalar advection
"""

import numpy as np
import matplotlib.pyplot as plt
import sandbox as sb

pi2 = 2.0*np.pi

# Parameters
nx  = 51
cfl = 0.5
T   = 2.0
nu  = 0.0075

# boundary conditions
bc0 = sb.general.fields.BoundaryCondition( name='periodic' )
bc1 = sb.general.fields.BoundaryCondition( name='periodic' )
# bc0 = sb.general.fields.BoundaryCondition( name='naive_adiabatic', indx= 0 )
# bc0 = sb.general.fields.BoundaryCondition( name='dirichlet',       indx= 0, val=2.0 )
# bc1 = sb.general.fields.BoundaryCondition( name='dirichlet',       indx=-1, val=2.0 )
# bc1 = sb.general.fields.BoundaryCondition( name='naive_outflow',   indx=-1 )

# waves travel across the domain in one period
c0 = 1.0
L  = 1.0*pi2
T *= L

# domain
x = np.linspace( 0, L, nx )
x = sb.fv.fields.Domain( x )
dx = min( x.dxh )

dt = cfl * dx / c0
nt = int( T / dt )

print('dif# =', nu*dt/(dx*dx))

# advected scalar
q = sb.fv.fields.UnsteadyField1D( 'q', x )
q.add_boundary_condition( bc1 )
q.add_boundary_condition( bc0 )
q.set_timestep( dt )
q.set_save_interval( nt=1 )

# initial conditions
q0 = 2.0 + np.sin( x.xp )
# q0 = 2.0 - x.xp
q.set_field( q0 )

# advection velocity
c = sb.fv.fields.Field1D( 'c', x )
c.set_field( c0*np.ones( len( x.xp ) ) )

# flux = sb.fv.advective_fluxschemes.CDS2()

flux = sb.fv.advective_fluxclasses.REAFlux1D()
flux.set_reconstruction( sb.fv.reconstructions.MC2() )
flux.set_evolution( sb.fv.evolutions.Upwind1() )

flux.set_advection_velocity( c )

# artificial smoothing if using CDS2 advective flux
smooth = sb.fv.diffusive_fluxschemes.CDS2()
smooth.set_diffusion_coefficient( nu )

# build equation
advEq = sb.fv.equationclass.Equation()
advEq.add_flux_term( flux   )
#advEq.add_flux_term( smooth )
advEq.set_time_integration( sb.fv.ODEintegrators.RungeKutta4 )

for n in range( nt ):
    advEq.step( q, dt )
    #bc0.val = 1.5 + 0.5*np.cos( pi2*q.t )

q.plot_history()

