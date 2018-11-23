"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Domain and Field classes for 1D scalar fields for Finite Volume solver
"""

import numpy as np

import maths_utils as mth
import timeseries_plotting as tsp
import sandbox as sb


class Domain( sb.general.fields.Domain ):
    """
        Domain of a finite volume field

        has attributes for location and spacing of mesh points
    """
    def __init__( self, mesh ):
        """
        initialise with cell size, and centre and vertex location, and number of cells
        """
        super( Domain, self ).__init__( mesh )

        self.xp = 0.5*( self.xh[:-1] + self.xh[1:] )

        self.dxp = np.diff( self.xp )

        return


class Field1D( sb.general.fields.Field1D ):
    """
        Class for 1D finite volume scalar fields

        field consists of array of value of the field, and list of boundary conditions
    """
    def __init__( self, name, mesh ):
        """
        initialise field with a name, mesh, and empty nodal value arrays
        """
        super( Field1D, self ).__init__( name, mesh )
        return


class UnsteadyField1D( Field1D ):
    """
        Class for unsteady 1D scalar fields

        same as Field1D, but value of field is saved at predetermined intervals in time
    """
    def __init__( self, name, mesh ):
        super( UnsteadyField1D, self ).__init__( name, mesh )
        self.dt = None
        self.nt = 0
        self.t  = 0
        self.save_interval = 1
        self.history = np.zeros( (1, len( self.val ) ) )
        return

    def set_field( self, data ):
        super(  UnsteadyField1D, self ).set_field( data )
        self.history[:] = self.val[:]

    def set_timestep( self, dt ):
        self.dt = dt
        return

    def set_save_interval( self, dt=None, nt=None ):
        if   dt == None and nt == None:
            return
        elif ( dt != None ) and ( nt != None ):
            print( 'UnsteadyField1D '+self.name+'.set_save_interval : cannot specify both dt and nt' )
            return
        elif dt== None:
            self.save_interval = nt
        elif nt == None:
            self.save_interval = int( round( dt / self.dt ) )

        return

    def copy( self ):
        """
        returns a copy (separate instance) of the field with identical mesh, nodal values, boundary conditions and time steps
        """
        new = super( UnsteadyField1D, self ).copy()
        new.set_timestep( self.dt )
        new.set_save_interval( nt=self.save_interval )
        new.t = self.t

        new.history = np.zeros_like( self.history )

        new.history[:,:] = self.history[:,:]

        return new

    def update( self, dvar ):
        super( UnsteadyField1D, self ).update( dvar )
        self.nt+=1
        self.t += self.dt

        if self.nt % self.save_interval == 0:
            self.history = np.append( self.history, [ self.val[:] ], axis=0 )

        return

    def plot_history( self ):
        tsp.view_timeseries1D( self )


class VectorField1D( object ):
    def __init__( self, n, name, mesh ):
        assert( len( name ) == n+1 )
        self.n     = n
        self.names = name
        self.name  = name[0]
        self.mesh  = mesh

        self.val = np.zeros( [ n, len( mesh.xp ) ] )

        self.q = []
        for i in range( n ):
            self.q.append( Field1D( name[i+1], mesh ) )
            self.q[i].val = self.val[i,:]

        return

    def set_field( self, data ):
        self.val[:,:] = data[:,:]
        return

    def copy( self ):
        g = VectorField1D( self.n, self.names, self.mesh )
        for i in range( self.n ):
            g.q[i] = self.q[i].copy()

        return g
        
    def add_boundary_condition( self, i, bc ):
        if type(i) == type(1):
            self.q[i].add_boundary_condition( bc )
            return

        elif type(i) == type('string'):
            for qi in self.q:
                if qi.name == i:
                    qi.add_boundary_condition( bc )
                    return

        else:
            print( 'invalid id for field of VectorField1D in add_boundary_condition' )
            return

    def update_ghosts( self ):
        pass

    def periodic( self, bc ):
        pass

    def __getitem__( self, index ):
        return self.q[index]

    def __add__( self, b ):
        assert self.mesh is b.mesh

        new = self.copy()
        new.set_field( self.val + b.val )
        return new

    def __sub__( self, b ):
        assert self.mesh is b.mesh

        new = self.copy()
        new.set_field( self.val - b.val )
        return new

    def __mul__( self, b ):
        new = self.copy()

        if( 'mesh' in dir(b) ):
            if( self.mesh is b.mesh ):
                new.set_field( self.val * b.val )
            else:
                return None
        else:
            new.set_field( self.val * b )

        return new

    def __div__( self, b ):
        new = self.copy()

        if( 'mesh' in dir(b) ):
            if( self.mesh is b.mesh ):
                new.set_field( self.val / b.val )
            else:
                return None
        else:
            new.set_field( self.val / b )

        return new

    def __truediv__( self, b ):
        new = self.copy()

        if( 'mesh' in dir(b) ):
            if( self.mesh is b.mesh ):
                new.set_field( self.val / b.val )
            else:
                return None
        else:
            new.set_field( self.val / b )

        return new

    def __eq__( self, b ):
        if self is b: return True

        if( type(b) != type( self ) ): return False

        if( self.n != b.n ): return False

        if( self.mesh is not b.mesh ): return False

        for i in range( self.n ):
            if( self[i] != b[i] ): return False

        return True

    def __ne__( self, b ):
        return ( ( self == b ) == False )

