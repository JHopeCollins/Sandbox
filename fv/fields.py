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

        self.xh_wg = np.append( self.xh, self.xh[-1] + self.dxh[-1] )
        self.xh_wg = np.append( self.xh[0] - self.dxh[0], self.xh_wg )

        self.dxh_wg = np.append( self.dxh, self.dxh[-1] )
        self.dxh_wg = np.append( self.dxh[0], self.dxh_wg )

        self.xp_wg = 0.5*( self.xh_wg[:-1] + self.xh_wg[1:] )

        self.dxp_wg = np.diff( self.xp_wg )

        self.xp  = self.xp_wg[1:-1]
        self.xh  = self.xh_wg[1:-1]
        self.dxh = self.dxh_wg[1:-1]
        self.dxp = self.dxp_wg[1:-1]

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
        self.val_wg = np.zeros_like(self.mesh.xp_wg)
        self.val    = self.val_wg[1:-1]
        return

    def set_field( self, data ):
        """
        sets the nodal values to the value of data, and updates the ghost cells
        """
        super( Field1D, self ).set_field( data )
        self.update_ghosts()
        return

    def update( self, dval ):
        """
        update the nodal values by the amount dval
        """
        super( Field1D, self ).update( dval )
        self.update_ghosts()
        return

    def add_boundary_condition( self, bc ):
        """
            Add an existing boundary condition to the field.

            if location of input boundary condition already has a boundary condition, will replace with new one

            input arguments:
            bc: must be BoundaryCondition instance
        """
        super( Field1D, self ).add_boundary_condition( bc )
        self.update_ghosts( )
        return

    def update_ghosts( self ):
        """
        update value of ghost cells at boundaries
        """
        for bc in self.bconds:
            update_func = getattr( self, bc.name )
            update_func( bc )
        return

    def periodic( self, bc):
        """
        set ghose cell values for periodic boundary condition
        """
        self.val_wg[ 0] = self.val[-1]
        self.val_wg[-1] = self.val[ 0]
        return

    def dirichlet( self, bc ):
        """
        set ghost cell value for dirichlet boundary condition
        """
        ghost = bc.indx
        in1 = mth.step_into_array( bc.indx, 1 )

        diff  = self.val_wg[in1] - bc.val

        self.val_wg[ghost] = bc.val - diff
        return

    def neumann( self, bc ):
        """
        set ghost cell value for neumann boundary condition
        """
        dn = bc.indx*2+1

        ghost = bc.indx
        in1 = mth.step_into_array( bc.indx, 1 )

        dx = self.mesh.dxp_wg[ghost]

        self.val_wg[ghost] = self.val_wg[in1] - dn*bc.val*dx
        return

    def naive_adiabatic( self, bc ):
        """
        set ghost cell value for neumann boundary condition
        """
        ghost = bc.indx
        in2 = mth.step_into_array( bc.indx, 2 )
        in1 = mth.step_into_array( bc.indx, 1 )

        self.val_wg[in1  ] = self.val_wg[in2]
        self.val_wg[ghost] = self.val_wg[in2]


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
        elif dt != None and nt != None:
            print( 'UnsteadyField1D '+self.name+'.set_save_interval : cannot specify both dt and nt' )
            return
        elif dt== None:
            self.save_interval = nt
        elif nt == None:
            self.save_interval = int( round( dt / self.dt ) )

        return

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

