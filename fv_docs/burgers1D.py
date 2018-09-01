"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Finite volume approximation to 1D Burgers equation
Analytical solution from Step 4 of Lorena A. Barba's CFD Python
https://github.com/barbagroup/CFDPython
"""

import numpy as np
import matplotlib.pyplot as plt
import maths_utils as mth

import advective_fluxschemes as afx
import diffusive_fluxschemes as dfx
import fields
import equationclass
import ODEintegrators

# set up initial and analytical solution
ufunc = mth.BurgersWave1D()

#set up problem
nx = 201+2
nt = 100
L  = 2.*np.pi
dx = L/(nx-1)
nu = 0.27
dt = 0.001
t  = 0

x  = np.linspace(-dx/2.0, L + dx/2.0, nx)
x  = fields.Domain( x )
u  = fields.UnsteadyField1D( 'u', x )
u.set_boundary_condition( name='periodic' )
u.set_timestep( dt )
u.set_save_interval( nt=5 )

#set initial solution
u_0 = np.asarray([ufunc(t, x0, nu) for x0 in x.x_noghost])

u.set_field( u_0 )

print('cfl  =', max(u_0)*dt/dx)
print('dif# =', nu*dt/(dx*dx))
print('Pe   =', max(u_0)*dx/nu)

fluxv = dfx.CDS2()
fluxi = afx.UDS1()

fluxi.set_mesh( x )
fluxi.set_advection_velocity( u )

fluxv.set_mesh( x )
fluxv.set_diffusion_coefficient( nu )

bgrs = equationclass.Equation()
bgrs.set_variable(  u     )
bgrs.add_flux_term( fluxi )
bgrs.add_flux_term( fluxv )
bgrs.set_time_integration( ODEintegrators.RungeKutta4 )

# timestepping
for n in range(nt):

    bgrs.step( dt )

    # flux = fluxi.apply() + fluxv.apply()
    # dudt = -np.diff(flux)/dx
    # du   = dudt*dt
    # u.update( du )

u_exact = np.asarray( [ ufunc( nt*dt, xi, nu ) for xi in x.x_noghost ] )

fig1, ax1 = mth.plot1Dsolution( x.x_noghost, u.val_noghost, u_0=u_0, u_e=u_exact )
fig1.show()

