"""
Written by: J Hope-Collins ( joshua.hope-collins@eng.ox.ac.uk )

Abstract Domain class for 1D scalar fields with unspecified numerical scheme
"""

import numpy as np

class Domain( object ):
    def __init__( self, xh ):
        """
        initialise Domain with cell vertices, cell sizes and number of cells
        """
        self.xh  = xh.copy()
        self.xp  = xh.copy()
        self.dxh = np.diff( self.xh )
        self.nh  = len( self.dxh )
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
    def __init__( self, name, mesh ):
        """
        initialise field with a name, mesh, empty nodal value array and boundary condition list
        """
        self.name = name
        self.mesh = mesh
        self.val  = np.zeros_like( self.mesh.xp )
        self.bconds = []
        return        

    def set_field( self, data ):
        """
        sets the nodal values to the value of data
        """
        self.val[:] = data[:]
        return

    def update( self, dval ):
        """
        update the nodal values by the amount dval
        """
        self.val[:] += dval[:]
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

