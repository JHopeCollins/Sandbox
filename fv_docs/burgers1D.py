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

# set up initial and analytical solution
ufunc = mth.BurgersWave1D()

#set up problem
nx = 201+2
nt = 50
L  = 2.*np.pi
dx = L/(nx-1)
nu = 0.27
dt = 0.001
t  = 0

x  = np.linspace(-dx/2.0, L + dx/2.0, nx)
x  = fields.Domain( x )
u  = fields.Field1D( 'u', x )
u.set_boundary_condition( name='periodic' )

#set initial solution
u_0 = np.asarray([ufunc(t, x0, nu) for x0 in x.x_noghost])

u.set_field( u_0 )

print('cfl  =', max(u_0)*dt/dx)
print('dif# =', nu*dt/(dx*dx))
print('Pe   =', max(u_0)*dx/nu)

fluxv = dfx.CDS2()
fluxi = afx.UDS1()

fluxi.set_variable( u )
fluxi.set_advection_velocity( u )

fluxv.set_variable( u )
fluxv.set_diffusion_coefficient( nu )

#--------------------------------------------------

flux_inv = np.empty(nx+1)
flux_vis = np.empty(nx+1)
flux     = np.empty(nx+1)

# timestepping
for n in range(nt):

    # calculate inviscid and viscous fluxes across each face
    flux_vis = fluxv.apply()
    flux_inv = fluxi.apply()
    flux     = flux_inv + flux_vis

    dudt = -np.diff(flux)/dx
    du   = dudt*dt

    u.update( du )

u_exact = np.asarray( [ ufunc( nt*dt, xi, nu ) for xi in x.x_noghost ] )

fig1, ax1 = mth.plot1Dsolution( x.x_noghost, u.val_noghost, u_0=u_0, u_e=u_exact )
fig1.show()

