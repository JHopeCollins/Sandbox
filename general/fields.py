"""
Written by: J Hope-Collins ( joshua.hope-collins@eng.ox.ac.uk )

Abstract Domain class for 1D scalar fields with unspecified numerical scheme
"""

from __future__ import division
import numpy as np
import sandbox as sb

class Domain( object ):
    def __init__( self, xh ):
        """
        initialise Domain with cell vertices, cell sizes and number of cells
        """
        super( Domain, self ).__init__( )
        self.xh  = xh.copy()
        self.xp  = xh.copy()
        self.dxh = np.diff( self.xh )
        self.dxp = np.diff( self.xp )
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
        super( BoundaryCondition, self ).__init__( )
        self.name = name
        self.indx = indx
        self.val  = val
        return

    def __eq__( self, b ):
        """
        test if boundary conditions are equal (same type, location and value)
        """
        if b is self: return True
        if self.name != b.name: return False
        if self.indx != b.indx: return False
        if self.val  != b.val:  return False
        return True

    def __ne__( self, b ):
        return not self == b


class Field1D( object ):
    def __init__( self, name, mesh ):
        """
        initialise field with a name, mesh, empty nodal value array and boundary condition list
        """
        super( Field1D, self ).__init__( )
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

    def copy( self ):
        """
        returns a copy (separate instance) of the field with identical mesh, nodal values and boundary conditions
        """
        g = type(self)( self.name, self.mesh )
        g.set_field( self.val )
        for bc in self.bconds:
            g.add_boundary_condition( bc )
        return g

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

    def __getitem__( self, index ):
        """
        treat field like an array to access the .val attribute
        """
        return self.val[ index ]
        
    def __add__( self, b ):
        """
        adding two fields over the same domain adds their values, maintaining boundary conditions of first field
        """
        assert self.mesh is b.mesh

        new = self.copy()
        new.set_field( self.val + b.val )

        return new

    def __sub__( self, b ):
        """
        subtracting two fields over the same domain subtracts their values, maintaining boundary conditions of first field
        """
        assert self.mesh is b.mesh

        new = self.copy()
        new.set_field( self.val - b.val )

        return new

    def __mul__( self, b ):
        """
        multiplying two fields over the same domain multiplies their values, maintaining boundary conditions of first field
        """
        new = self.copy()

        if 'mesh' in dir( b ):
            if self.mesh is b.mesh:
                new.set_field( self.val * b.val )
            else:
                assert 0==1
        else:
            new.set_field( self.val * b )

        return new

    def __truediv__( self, b ):
        """
        dividing two fields over the same domain dividing their values, maintaining boundary conditions of first field
        """
        new = self.copy()

        if 'mesh' in dir( b ):
            if self.mesh is b.mesh:
                new.set_field( self.val / b.val )
            else:
                assert 0==1
        else:
            new.set_field( self.val / b )

        return new

    def __div__( self, b ):
        """
        dividing two fields over the same domain dividing their values, maintaining boundary conditions of first field
        """
        new = self.copy()

        if 'mesh' in dir( b ):
            if self.mesh is b.mesh:
                new.set_field( self.val / b.val )
            else:
                assert 0==1
        else:
            new.set_field( self.val / b )

        return new
        
    def __eq__( self, b ):
        """
        test if two fields are equivalent (same mesh, value and boundary conditions)
        """
        if b is self: return True

        if type(b) != type( self ): return False

        if self.mesh is not b.mesh: return False

        if np.any( self.val != b.val ): return False

        if len( self.bconds ) != len( b.bconds ): return False

        for bc0 in self.bconds:
            match = False
            for bc1 in b.bconds:
                if bc0 == bc1:
                    match = True
                    break
            if not match: return False

        return True
        
    def __ne__( self, b ):
        return not self == b

