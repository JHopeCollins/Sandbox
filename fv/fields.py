"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Domain and Field classes for 1D scalar fields for Finite Volume solver
"""

import numpy as np
import maths_utils as mth
import timeseries_plotting as tsp


class Domain( object ):
    """
        Domain of a finite volume field

        has attributes for location and spacing of mesh points
    """
    def __init__( self, x ):
        self.set_x( x )
        return

    def set_x( self, x ):
        self.x  = x.copy()
        self.dx = np.diff( x )
        self.h  = ( self.dx[:-1] + self.dx[1:] )/2.0

        self.h = np.append( np.append( self.h[0], self.h ), self.h[1] )

        self.x_noghost  = self.x[ 1:-1]
        self.dx_noghost = self.dx[1:-1]
        return


class BoundaryCondition( object ):
    """
        Class for a boundary condition for a field

        boundary condition has attributes:
        name: string, name of flux class method which applies the boundary condition.
        indx: (0 or -1) index of boundary to apply condition to.
        val : value of boundary condition, eg for dirichlet condition
    """
    def __init__( self, name=None, indx=None, val=None ):
        self.name = name
        self.indx = indx
        self.val  = val
        return


class Field1D( object ):
    """
        Class for 1D finite volume scalar fields

        field consists of array of value of the field, and list of boundary conditions
    """

    def __init__( self, name, mesh ):
        self.name   = name
        self.mesh   = mesh
        self.val    = np.zeros_like(self.mesh.x)
        self.val_noghost = self.val[1:-1]
        self.bconds = []
        return

    def set_field( self, data ):
        self.val_noghost[:] = data[:]
        self.update_ghosts()
        return

    def copy( self ):
        g = Field1D( self.name, self.mesh )
        g.set_field( self.val_noghost )
        for bc in self.bconds:
            g.add_boundary_condition( bc )
        return g

    def update( self, dvar ):
        self.val_noghost[:] += dvar[:]
        self.update_ghosts()
        return

    def add_boundary_condition( self, bc ):
        """
            Add an existing boundary condition to the field.

            if location of input boundary condition already has a boundary condition, will replace with new one

            input arguments:
            bc: must be BoundaryCondition instance
        """
        identifier = 'boundary condition '+bc.name+' on field '+self.name

        if bc.name == 'periodic':

            del self.bconds[:]

        else:

            if bc.indx == None:
                print( 'bc index must be specified for non-periodic '+identifier )
                return
            if ( bc.indx != 0 ) and (bc.indx != -1 ):
                print( 'bc index must be 0 or -1 for non-periodic '+identifier )
                return

            for i, b in enumerate( self.bconds ):
                if ( bc.indx == b.indx ) or ( b.name == 'periodic' ):
                    del self.bconds[i]
                    break

        self.bconds.append( bc )
        self.update_ghosts( )

        return

    def set_boundary_condition( self, name, indx=None, val=None ):
        """
            set a boundary condition to the field.

            if location of input boundary condition already has a boundary condition, will replace with new one

            input arguments:
            name: string name of boundary condition
            indx: for non-periodic bc, 0 or -1 for beginning/end of domain
            val : for non-periodic bc, eg derivative value for neumann
        """
        bc = BoundaryCondition( name, indx, val )
        self.add_boundary_condition( bc )
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
        self.val[ 0] = self.val[-2]
        self.val[-1] = self.val[ 1]
        return

    def dirichlet( self, bc ):
        """
        set ghost cell value for dirichlet boundary condition
        """
        ghost = bc.indx
        in1 = mth.step_into_array( bc.indx, 1 )

        diff  = self.val[in1] - bc.val

        self.val[ghost] = bc.val - diff
        return

    def neumann( self, bc ):
        """
        set ghost cell value for neumann boundary condition
        """
        dn = bc.indx*2+1

        ghost = bc.indx
        in1 = mth.step_into_array( bc.indx, 1 )

        dx = self.mesh.dx[ghost]

        self.val[ghost] = self.val[in1] - dn*bc.val*dx
        return


class UnsteadyField1D( Field1D ):
    """
        Class for unsteady 1D scalar fields

        same as Field1D, but value of field is saved at predetermined intervals in time
    """
    def __init__( self, name, mesh ):
        super( self.__class__, self ).__init__( name, mesh )
        self.dt = None
        self.nt = 0
        self.t  = 0
        self.save_interval = 1
        self.history = np.zeros( (1, len( self.val_noghost ) ) )
        return

    def set_field( self, data ):
        super(  self.__class__, self ).set_field( data )
        self.history[:] = self.val_noghost[:]

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
        super( self.__class__, self ).update( dvar )

        self.nt+=1
        self.t += self.dt

        if self.nt % self.save_interval == 0:
            self.history = np.append( self.history, [self.val_noghost], axis=0 )

        return

    def plot_history( self ):
        tsp.view_timeseries1D( self )


