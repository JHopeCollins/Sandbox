"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Finite volume approximation to 1D Burgers equation
Analytical solution from Step 4 of Lorena A. Barba's CFD Python
https://github.com/barbagroup/CFDPython
"""

import numpy as np
import matplotlib.pyplot as plt
import maths_utils as mth
import sandbox as sb

import advective_fluxclasses as afc
import advective_fluxschemes as afs
import diffusive_fluxschemes as dfs
import fields
import equationclass
import ODEintegrators
import reconstructions as rc
import evolutions as ev

# set up initial and analytical solution
ufunc = mth.BurgersWave1D()

#set up problem
nx = 201
nt = 1000
L  = 2.*np.pi
dx = L/(nx-1)
nu = 0.17
dt = 0.001
t  = 0

x  = np.linspace(0, L, nx)
x  = fields.Domain( x )
u  = fields.UnsteadyField1D( 'u', x )
u.set_timestep( dt )
u.set_save_interval( nt=5 )

bcp  = sb.general.fields.BoundaryCondition( name='periodic' )

# bcn0 = sb.general.fields.BoundaryCondition( name='naive_adiabatic', indx= 0, val=0 )
bcn0 = sb.general.fields.BoundaryCondition( name='dirichlet', indx= 0, val= 5.0 )
# bcn0 = sb.general.fields.BoundaryCondition( name='neumann',   indx= 0, val= 0.2 )

bcn1 = sb.general.fields.BoundaryCondition( name='naive_outflow',   indx=-1, val=0 )

# u.add_boundary_condition( bcp  )
u.add_boundary_condition( bcn0 )
u.add_boundary_condition( bcn1 )

#set initial solution
u_0 = np.asarray([ufunc(t, x0, nu) for x0 in x.xp])

u.set_field( u_0 )

print('cfl  =', max(u_0)*dt/dx)
print('dif# =', nu*dt/(dx*dx))
print('Pe   =', max(u_0)*dx/nu)

# fluxi = afs.CDS2()
# fluxi.set_advection_velocity( u )

fluxi = afc.REAFlux1D()
fluxi.set_reconstruction( rc.PCM1() )
fluxi.set_evolution( ev.Upwind1() )
fluxi.set_advection_velocity( u )

fluxv = dfs.CDS2()
fluxv.set_diffusion_coefficient( nu )

bgrs = equationclass.Equation()
bgrs.add_flux_term( fluxi )
#bgrs.add_flux_term( fluxv )
bgrs.set_time_integration( ODEintegrators.RungeKutta4 )

# timestepping
for n in range( nt ):
    bgrs.step( u, dt )

u.plot_history()

