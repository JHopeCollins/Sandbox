"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Finite volume approximation to 1D Burgers equation
Analytical solution from Step 4 of Lorena A. Barba's CFD Python
https://github.com/barbagroup/CFDPython
"""

import numpy as np
import matplotlib.pyplot as plt
import maths_utils as mth

import fluxclass
import fluxes1Dinv as f1i
# import fluxes1Dvis as f1v

# set up initial and analytical solution
ufunc = mth.BurgersWave1D()

#set up problem
nx = 101
nt = 300
L  = 2.*np.pi
dx = L/(nx-1)
nu = .17
dt = 0.001

x = np.linspace(dx/2.0, L - dx/2.0, nx)
un = np.empty(nx)
t = 0

#set initial solution
u_0 = np.asarray([ufunc(t, x0, nu) for x0 in x])
u = u_0.copy()

print('cfl  =', max(u_0)*dt/dx)
print('dif# =', nu*dt/(dx*dx))
print('Pe   =', max(u_0)*dx/nu)

#--------------------------------------------------

fluxv = fluxclass.vCDS4()
fluxv.set_boundary_condition('periodic')

fluxi = fluxclass.iUDS1()
fluxi.set_boundary_condition('periodic')

#--------------------------------------------------

flux_inv = np.empty(nx+1)
flux_vis = np.empty(nx+1)
flux     = np.empty(nx+1)

# timestepping
for n in range(nt):
    un = u.copy()

    # calculate inviscid and viscous fluxes across each face
    arglist_vis = [u, dx, nu]
    arglist_inv = [u, u]
    flux_vis = fluxv.apply(arglist_vis)
    flux_inv = fluxi.apply(arglist_inv)

    flux = (flux_inv + flux_vis)

    dudt = -np.diff(flux_inv + flux_vis)/dx

    # update cell values
    u = un + dudt*dt

u_exact = np.asarray([ufunc(nt*dt, xi, nu) for xi in x])

fig1, ax1 = mth.plot1Dsolution(x, un, u_0=u_0, u_e=u_exact)
fig1.show()

