"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Field class for 1D scalar fields
"""

import numpy as np


class Domain ( object ):
    """
        Domain of a field

        has attributes for location and spacing of mesh points
    """
    def __init__( self, x ):
        """
            Create Domain instance with mesh point locations and spacing
        """
        self.set_x( x )
        return

    def set_x( self, x ):
        self.x  = x
        self.dx = np.diff( x )
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
        """
            Create BoundaryCondition instance with name, location and value attributes
        """
        self.name = name
        self.indx = indx
        self.val  = val
        return


class Field1D( object ):
    """
        Class for 1D scalar fields

        field consists of array of value of the field, and list of boundary conditions
    """

    def __init__( self, name, mesh ):
        """
        Create field instance
        """
        self.name   = name
        self.mesh   = mesh
        self.val    = np.zeros_like(self.mesh.x)
        self.bconds = []

        return

    def set_field( self, data ):
        self.val[1:-1] = data[:]
        self.update_ghosts
        return

    def update( self, update ):
        self.val[:] += update[:]
        return

    def set_boundary_condition( self, name=None, indx=None, val=None ):
        """
            Add a boundary condition to the field.

            if location of input boundary condition already has a boundary condition, will replace with new one

            input arguments:
            bc: must be BoundaryCondition instance
        """
        bc = BoundaryCondition( name, indx, val )

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
        first = mth.step_into_array( bc.indx, 1 )

        diff  = self.val[first] - bc.val

        self.val[ghost] = bc.val - diff
        return

    def neumann( self, bc ):
        """
        set ghost cell value for neumann boundary condition
        """
        ghost = bc.indx
        first = mth.step_into_array( bc.indx, 1 )

        dx = self.mesh.dx[ghost]

        self.val[ghost] = self.val[first] + bc.val*dx
        return

