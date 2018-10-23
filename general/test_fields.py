"""
Written by: J Hope-Collins ( joshua.hope-collins@eng.ox.ac.uk )

Test suite for general/fields.py file
"""

import numpy as np
import fields


class Test_Domain( object ):
    def test_init( self ):
        mesh = np.linspace( 0.0, 1.0, 11 )
        d = fields.Domain( mesh )

        assert d.xh is not mesh
        assert np.all( d.xh  ==          mesh )
        assert np.all( d.dxh == np.diff( mesh ) )
        assert d.nh == 10

        return


class Test_BoundaryCondition( object ):
    def test_init( self ):
        bc = fields.BoundaryCondition( )
        assert bc.name == None
        assert bc.indx == None
        assert bc.val  == None

        bc = fields.BoundaryCondition( name='periodic' )
        assert bc.name == 'periodic'
        assert bc.indx == None
        assert bc.val  == None

        bc = fields.BoundaryCondition( name='periodic', indx=0, val=1 )
        assert bc.name == 'periodic'
        assert bc.indx == 0
        assert bc.val  == 1

        return


class Test_Field1D( object ):
    def test_init( self ):
        x = np.linspace( 0, 1, 11 )
        x = fields.Domain( x )
        f = fields.Field1D( 'q', x )

        assert f.mesh is x
        assert f.name   == 'q'
        assert f.bconds == [ ]
        assert len( f.val ) == len( f.mesh.xp )
        assert np.all( f.val == 0 )

        return

    def test_set_field( self ):
        x = np.linspace( 0, 1, 11 )
        x = fields.Domain( x )
        f = fields.Field1D( 'q', x )

        data = np.asarray( range( len( x.xp ) ) )
        f.set_field( data )

        assert np.all( f.val == data )
        assert f.val is not data

        return

    def test_update( self ):
        x = np.linspace( 0, 1, 11 )
        x = fields.Domain( x )
        f = fields.Field1D( 'q', x )

        data = np.asarray( range( len( x.xp ) ) )
        f.set_field( data )
        f.update(    data )

        assert np.all( f.val == 2*data )

        return

    def test_add_boundary_condition( self ):
        mesh = np.linspace( -0.05, 1.05, 12 )
        d = fields.Domain( mesh )
        f = fields.Field1D( 'x', d )

        bcp = fields.BoundaryCondition( name='periodic' )
        bcd = fields.BoundaryCondition( name='dirichlet', indx=0, val=0 )
        bcn = fields.BoundaryCondition( name='neumann',  indx=-1, val=1 )

        f.add_boundary_condition( bcp )
        assert bcp in f.bconds
        assert len( f.bconds ) == 1

        f.add_boundary_condition( bcd )
        assert bcp not in f.bconds
        assert bcd in f.bconds
        assert len( f.bconds ) == 1

        f.add_boundary_condition( bcn )
        assert bcn, bcd in f.bconds
        assert len( f.bconds ) == 2

        bcn2 = fields.BoundaryCondition( name='neumann',  indx=0, val=1 )
        f.add_boundary_condition( bcn2 )
        assert bcd not in f.bconds
        assert bcn2    in f.bconds

        f.add_boundary_condition( bcp )
        assert bcn, bcn2 not in f.bconds
        assert bcp           in f.bconds

        bcn.indx=None
        f.add_boundary_condition( bcn )
        assert bcn not in f.bconds
        assert bcp     in f.bconds

        bcn.indx=3
        f.add_boundary_condition( bcn )
        assert bcn not in f.bconds
        assert bcp     in f.bconds

        return

    def test_set_boundary_condition( self ):
        mesh = np.linspace( -0.05, 1.05, 12 )
        d = fields.Domain( mesh )
        f = fields.Field1D( 'x', d )

        f.set_boundary_condition( 'periodic' )

        assert type( f.bconds[0] ) == fields.BoundaryCondition
        assert f.bconds[0].name == 'periodic'
        assert f.bconds[0].indx == None
        assert f.bconds[0].val  == None

        f.set_boundary_condition( name='neumann', indx=0, val=1 )

        assert f.bconds[0].name == 'neumann'
        assert f.bconds[0].indx == 0
        assert f.bconds[0].val  == 1

        return

