"""
Written by: J Hope-Collins ( joshua.hope-collins@eng.ox.ac.uk )

classes for 1D Euler equations
"""

import numpy as np

import maths_utils as mth
import fields
import advective_fluxclasses as afc


class u( object ):
    def __init__( self, q ):
        self.q = q
        return

    def __getitem__( self, index ):
        """
        return velocity: u = rho*u / rho
        """
        return self.q.rhou[index] / self.q.rho[index]


class E( object ):
    def __init__( self, q ):
        self.q = q
        return

    def __getitem__( self, index ):
        """
        return total energy: E = rho*E / rho
        """
        return self.q.rhoe[index] / self.q.rho[index]


class e( object ):
    def __init__( self, q ):
        self.q = q
        return

    def __getitem__( self, index ):
        """
        return specific energy: e = E - 0.5*u^2
        """
        return ( self.q.rhoe[index] - 0.5*self.q.rhou[index]*self.q.u[index] ) / self.q.rho[index]


class T( object ):
    def __init__( self, q ):
        self.q = q
        return

    def __getitem__( self, index ):
        """
        return temperature: T = e / Cv
        """
        cv1 = 1. / self.q.Cv
        return self.q.e[index] * cv1


class p( object ):
    def __init__( self, q ):
        self.q = q
        return

    def __getitem__( self, index ):
        """
        return pressure: P = rho * R * T
        """
        return self.q.rho[index] * self.q.R * self.q.T[index]


class H( object ):
    def __init__( self, q ):
        self.q = q
        return

    def __getitem__( self, index ):
        """
        return total enthalpy: H = (rho*E + p)/rho
        """
        return ( self.q.rhoe[index] + self.q.p[index] ) / self.q.rho[index]


class a( object ):
    def __init__( self, q ):
        self.q = q
        return

    def __getitem__( self, index ):
        """
        return speed of sound: a = sqrt( gamma * R * T )
        """
        return np.sqrt( self.q.gamma * self.q.R * self.q.T[index] )


class M( object ):
    def __init__( self, q ):
        self.q = q
        return

    def __getitem__( self, index ):
        """
        return local Mach number: M = u/a
        """
        return self.q.u[index] / self.q.a[index]


class EulerField1D( fields.VectorField1D ):
    def __init__( self, mesh ):
        names = [ 'q', 'rho', 'rhou', 'rhoe' ]
        super( EulerField1D, self ).__init__( 3, names, mesh )
        # self.rho  = self.q[0]
        # self.rhou = self.q[1]
        # self.rhoe = self.q[2]

        self.gamma = 1.4
        self.R     = 287.
        self.Cp    = self.gamma*self.R/ ( self.gamma-1. )
        self.Cv    = self.Cp / self.gamma

        self.u = u( self )
        self.E = E( self )
        self.e = e( self )
        self.T = T( self )
        self.p = p( self )
        self.H = H( self )
        self.a = a( self )
        self.M = M( self )

        return

    def copy( self ):
        g = type(self)( self.mesh )
        for i in range( self.n ):
            g.q[i] = self.q[i].copy()
            g.q[i].val = g.val[i,:]

        for bc in self.bconds:
            g.add_boundary_condition( bc )

        return g


class EulerUDS1( afc.VectorUpwindFlux1D ):
    def __init__( self, q ):
        super( EulerUDS1, self ).__init__()
        self.stencil_radius = 1

        self.vel = q.u
        return

    def set_advection_velocity( self ): pass

    def arg_list( self, q ):
        args = []

        f = np.zeros( q.val.shape )
        f[0,:] = q.rhou[:]
        f[1,:] = q.rhou[:]*q.u[:] + q.p[:]
        f[2,:] = q.rhou[:]*q.H[:]

        args.append( q.u[:] )
        args.append( f )

        return args

    def flux_calculation( self, args ):
        u  = args[0]
        f  = args[1]

        upwind = self.get_upwind_indx( u, 1 )

        flux = f[:,upwind]

        return flux

