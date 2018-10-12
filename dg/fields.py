"""
Written by: J Hope-Collins ( joshua.hope-collins@eng.ox.ac.uk )

Domain and Field classes for 1D scalar fields for Discontinuous Galerkin solver
"""

import numpy as np
import polys
import timeseries_plotting as tsp

class Domain( object ):
    """
        Domain for a discontinuous galerkin field

        has attributes for cell vertex and nodal points, quadrature points and weights, cell boundary extrapolation vectors and cell jacobians, and mass matrix
    """
    def __init__( self, xh ):
        """
        initialise Domain with cell vertices and sizes
        """
        self.xh = xh                    # cell vertex points
        self.dx = np.diff( self.xh )    # cell size
        self.nh = len(     self.dx )    # number of cells
        self.dg = 0.5*self.dx           # cell jacobian
        return

    def set_expansion( self, order ):
        """
        set the expansion basis type and order, and the quadrature point type

        calculate the physical nodal points, cell jacobians and cell boundary extrapolation vectors
        """
        self.p = order

        q = polys.Legendre()
        self.zp = q.zeros[   order ]
        self.wp = q.weights[ order ]

        b = polys.Lagrange()
        b.interpolate( self.zp, np.ones( self.p ) )

        self.Lm1 = np.zeros( self.nh*self.p )   #extrapolation to left  cell face
        self.Lp1 = np.zeros( self.nh*self.p )   #extrapolation to right cell face
        self.xp  = np.zeros( self.nh*self.p )   # nodal points
        self.M   = np.zeros( self.nh*self.p )   # mass matrix

        tLm1 = map( lambda j : b.ljx( j, -1 ), range( self.p ) )
        tLp1 = map( lambda j : b.ljx( j,  1 ), range( self.p ) )

        for k in range( self.nh ):
            kL = self.p*k
            kR = self.p + kL

            g = polys.g1D( self.xh[k], self.xh[k+1] )

            self.M[   kL : kR ] = self.dg[k]*self.wp
            self.xp[  kL : kR ] = g( self.zp )
            self.Lm1[ kL : kR ] = tLm1
            self.Lp1[ kL : kR ] = tLp1

        return


class Field1D( object ):
    """
        class for 1D discontinuous galerkin field

        field consists of array of nodal values, mesh, and list of boundary conditions
    """
    def __init__( self, name, mesh ):
        """
        initialise field with a name, mesh, and empty nodal value array
        """
        self.name   = name
        self.mesh   = mesh
        self.p      = mesh.p
        self.bconds = []
        self.val    = np.zeros_like( self.mesh.xp )
        return

    def set_field( self, data ):
        """
        sets the nodal values to the value of data
        """
        self.val[:] = data[:]
        return

    def copy( self ):
        """
        returns a copy (separate instance) of the field with identical mesh, nodal values and boundary conditions
        """
        g = Field1D( self.name, self.mesh )
        g.set_field( self.val )
        for bc in self.bconds:
            g.bconds.append( bc )

        return g

    def update( self, dval ):
        """
        update the nodal values by the amount dval
        """
        self.val[:] += dval[:]
        return


class UnsteadyField1D( Field1D ):
    """
        class for unsteady 1D discontinuous galerkin field

        same as Field1D, but value of field is saved at predetermined intervals in time
    """
    def __init__( self, name, mesh ):
        super( self.__class__, self ).__init__( name, mesh )
        self.dt = None
        self.nt = 0
        self.t  = 0
        self.save_interval = 1
        self.history = np.zeros( (1, len( self.val ) ) )
        return

    def set_field( self, data ):
        """
        set the value of the field and start recording history
        """
        super(  self.__class__, self ).set_field( data )
        self.history[:] = self.val[:]
        return

    def set_timestep( self, dt ):
        self.dt = dt
        return

    def set_save_interval( self, dt=None, nt=None ):
        if   dt == None and nt == None:
            return
        elif dt != None and nt != None:
            print( 'UnsteadyField1D '+self.name+'.set_save_interval : cannot specify both dt and nt' )
            return
        elif dt== None:
            self.save_interval = nt
        elif nt == None:
            self.save_interval = int( round( dt / self.dt ) )

        if self.save_interval == 0:
            self.save_interval = 1
        return

    def update( self, dvar ):
        """
        update nodal value and save periodically save history
        """
        super( self.__class__, self ).update( dvar )

        self.nt+=1
        self.t += self.dt

        if self.nt % self.save_interval == 0:
            self.history = np.append( self.history, [self.val], axis=0 )

        return

    def plot_history( self ):
        tsp.view_timeseries1D( self )





