"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Domain and Field classes for 1D scalar fields for Finite Volume solver
"""

import numpy as np

import maths_utils as mth
import timeseries_plotting as tsp
from sandbox import general


class Domain( general.fields.Domain ):
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


class Field1D( general.fields.Field1D ):
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

    def copy( self ):
        """
        returns a copy (separate instance) of the field with identical mesh, nodal values and boundary conditions
        """
        g = Field1D( self.name, self.mesh )
        g.set_field( self.val)
        for bc in self.bconds:
            g.add_boundary_condition( bc )
        return g

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


