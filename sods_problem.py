"""
Written by: Josh Hope-Collins ( joshua.hope-collins@eng.ox.ac.uk )

simulation of sod's shock tube problem
"""

import numpy as np
import matplotlib.pyplot as plt

import sandbox as sb

# function: equation of state
def IdealPfromRhoTemp( pressure, rho, temperature ):
    """
    update pressure according to P/rho = RT
    """
    R = 287.0
    pressure.val_wg[:] = R*rho.val_wg[:]*temperature.val_wg[:]
    return

def IdealTfromPRho( T, pressure, density ):
    R = 287.0
    T.val_wg[:] = p.val_wg[:] / ( R*rho.val_wg[:] )
    return
    
def IdealEfromPRhoU( E, p, rho, u ):
    E.val_wg[:] = p.val_wg[:]*(1.0/0.4) + 0.5*rho.val_wg[:]*u.val_wg[:]*u.val_wg[:]
    return

def IdealPfromERhoU( P, E, rho, u ):
    p.val_wg[:] = 0.4*( E.val_wg[:] - 0.5 * rho.val_wg[:] * u.val_wg[:] * u.val_wg[:] )
    return


# domain and boundary conditions
x = np.linspace( -1, 1, 101 )
x = sb.fv.fields.Domain( x )

time = 0.2
cfl  = 0.1
dt   = cfl*min(x.dxh)
nt   = int( time / dt )
nt   = 10

bc0 = sb.general.fields.BoundaryCondition( name='naive_adiabatic', indx= 0, val=0 )
bc1 = sb.general.fields.BoundaryCondition( name='naive_adiabatic', indx=-1, val=0 )

# conserved variables
#   density
#   velocity
#   energy

rho = sb.fv.fields.UnsteadyField1D( 'density',  x )
mom = sb.fv.fields.UnsteadyField1D( 'velocity', x )
E   = sb.fv.fields.UnsteadyField1D( 'energy',   x )

rho.set_timestep( dt )
mom.set_timestep( dt )
E.set_timestep(   dt )

rho.add_boundary_condition( bc0 )
rho.add_boundary_condition( bc1 )
mom.add_boundary_condition( bc0 )
mom.add_boundary_condition( bc1 )
E.add_boundary_condition( bc0 )
E.add_boundary_condition( bc1 )

# secondary variables
#   pressure
#   pressure*velocity

p = sb.fv.fields.Field1D( 'pressure',    x )
u = sb.fv.fields.Field1D( 'velocity',    x )

p.add_boundary_condition( bc0 )
p.add_boundary_condition( bc1 )

# initial conditions
#   density
#   velocity
#   energy
ones = np.ones( 100 )
rhoL, rhoR = 1.0, 0.125
uL,   uR   = 0.0, 0.0
pL,   pR   = 1.0, 0.1

temp = np.append( ones[:50]*rhoL, ones[50:]*rhoR )
rho.set_field( temp )
temp = np.append( ones[:50]*uL, ones[50:]*uR )
u.set_field( temp )
temp = np.append( ones[:50]*pL, ones[50:]*pR )
p.set_field( temp )

pu = p*u
IdealEfromPRhoU( E, p, rho, u )

E.set_field( E.val )
mom.set_field( (rho*u)[:] )

# equations
#   density
#   velocity
#   energy
rhoEq = sb.fv.equationclass.Equation()
momEq = sb.fv.equationclass.Equation()
EEq   = sb.fv.equationclass.Equation()

rhoEq.set_variable( rho )
momEq.set_variable( mom )
EEq.set_variable( E )

# density fluxes
#   advection
rho_advection = sb.fv.advective_fluxschemes.UDS1()
rho_advection.set_mesh( x )
rho_advection.set_advection_velocity( u )

rhoEq.add_flux_term( rho_advection )

# velocity fluxes
#   advection
#   pressure
mom_advection = sb.fv.advective_fluxschemes.UDS1()
mom_advection.set_mesh( x )
mom_advection.set_advection_velocity( u )

mom_pressure = sb.fv.advective_fluxschemes.PressureUDS1()
mom_pressure.set_mesh( x )
mom_pressure.set_advection_velocity( u )
mom_pressure.set_pressure( p )

momEq.add_flux_term( mom_advection )
momEq.add_flux_term( mom_pressure  )

# energy fluxes
#   advection
#   pressure
E_advection = sb.fv.advective_fluxschemes.UDS1()
E_advection.set_mesh( x )
E_advection.set_advection_velocity( u )

E_pressure = sb.fv.advective_fluxschemes.PressureUDS1()
E_pressure.set_mesh( x )
E_pressure.set_advection_velocity( u )
E_pressure.set_pressure( pu )

EEq.add_flux_term( E_advection )
EEq.add_flux_term( E_pressure  )

# timestepping
#   density
#   velocity
#   energy
rhoEq.set_time_integration( sb.fv.ODEintegrators.RungeKutta4 )
momEq.set_time_integration( sb.fv.ODEintegrators.RungeKutta4 )
EEq.set_time_integration(   sb.fv.ODEintegrators.RungeKutta4 )

for n in range( nt ):
    u.set_field( (mom/rho)[:] )
    IdealPfromERhoU( p, E, rho, u )
    pu.set_field( (p*u)[:] )

    rhoEq.calculate_update(  dt )
    momEq.calculate_update(  dt )
    EEq.calculate_update(    dt )

    rhoEq.apply_update()
    momEq.apply_update()
    EEq.apply_update()

rho.plot_history()
mom.plot_history()
E.plot_history()
"""
"""
