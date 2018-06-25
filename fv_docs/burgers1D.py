"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Finite volume approximation to 1D Burgers equation
Analytical solution from Step 4 of Lorena A. Barba's CFD Python
https://github.com/barbagroup/CFDPython

Numerical scheme:
    Spatial (convective): 2nd order CDS
    Spatial (diffusive ): 2nd order CDS
    Temporal: 1st order explicit Euler
"""

import numpy as np
import sympy
import matplotlib.pyplot as plt

# set up initial and analytical solution
x, nu, t = sympy.symbols('x nu t')
phi = (sympy.exp(-(x - 4*t)**2 / (4*nu*(t+1))) + \
       sympy.exp(-(x - 4*t - 2*np.pi)**2 / (4*nu*(t+1))))

phiprime = phi.diff(x)

u = -2*nu*(phiprime / phi) + 4
ufunc = sympy.utilities.lambdify((t, x, nu), u)

#set up problem
nx = 101
nt = 10
L  = 2.*np.pi
dx = L/(nx-1)
nu = .50
dt = 0.001

x = np.linspace(0, L, nx)
un = np.empty(nx)
t = 0

#set initial solution
u_0 = np.asarray([ufunc(t, x0, nu) for x0 in x])
u_0[-1] = u_0[0]
u = u_0.copy()

flux_inv = np.empty(nx-1)
flux_vis = np.empty(nx-1)
flux     = np.empty(nx-1)

# timestepping
for n in range(nt):
    un = u.copy()

    # calculate inviscid and viscous fluxes across each face
    flux_inv =    (un[1:] + un[:-1])*(un[1:] + un[:-1])/4.0
    flux_vis = nu*(un[1:] - un[:-1])/dx

    flux = (flux_inv + flux_vis)

    # update cell values
    u[:-1] = un[:-1] - flux*dt/dx
    u[1:]  = u[1:]   + flux*dt/dx

    # periodic boundary conditions
    u[0] = u[0] + u[-1] - un[0]
    u[-1] = u[0]

u_exact = np.asarray([ufunc(nt*dt, xi, nu) for xi in x])

fig1, ax1 = plt.subplots(1, 1)
ax1.plot(x, u_0, label='initial')
ax1.plot(x, u_exact, label='exact')
ax1.plot(x, u, label='numerical')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$u$')
ax1.legend()
fig1.show()
