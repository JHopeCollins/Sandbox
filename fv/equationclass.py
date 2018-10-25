"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Equation class for finite volume discretisations of homogeneous hyperbolic equation
"""

import numpy as np

class Equation( object ):
    """
    Equation class for finite volume discretisations of homogeneous hyperbolic equation

    Contains a variable being transported, list of flux instances for each term in the equation, and an ODE integrator for timestepping
    """
    def __init__( self ):
        self.flux_terms = []
        return

    def set_variable( self, q ):
        self.q  = q
        self.mesh = q.mesh
        return

    def set_time_integration( self, ODEintegrator ):
        self.time_stepper = ODEintegrator
        return

    def add_flux_term( self, flux):
        self.flux_terms.append( flux )
        return

    def spatial_operator( self, q ):
        flux = np.zeros( len( q.val ) +1 )

        for term in self.flux_terms:
            flux[:] += term.apply( q )

        return np.diff( flux ) / q.mesh.dxh

    def step(self, dt ):
        dq = self.time_stepper( dt, self.q, self.spatial_operator )
        self.q.update( dq )
        return

