
"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Finite volume approximation to 1D Burgers equation
Analytical solution from Step 4 of Lorena A. Barba's CFD Python
https://github.com/barbagroup/CFDPython
"""

import numpy as np
import maths_utils as mth

import advective_fluxschemes as afx
import diffusive_fluxschemes as dfx
import fields

def burgers( nt, dt=0.001, nx = 203, nu=0.27, save_interval=5 ):

    ufunc = mth.BurgersWave1D()

    #set up problem
    L  = 2.*np.pi
    dx = L/(nx-1)
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

    return u

