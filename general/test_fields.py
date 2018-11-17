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

    def test__eg__( self ):
        bc0 = fields.BoundaryCondition( )
        bc1 = fields.BoundaryCondition( )
        assert bc0 == bc0
        assert bc0 == bc1

        bc0 = fields.BoundaryCondition( name='periodic' )
        assert bc0 != bc1
        bc1 = fields.BoundaryCondition( name='periodic' )
        assert bc0 == bc1

        bc0 = fields.BoundaryCondition( name='periodic', indx=0 )
        assert bc0 != bc1
        bc1 = fields.BoundaryCondition( name='periodic', indx=0 )
        assert bc0 == bc1

        bc0 = fields.BoundaryCondition( name='periodic', indx=0, val=0 )
        assert bc0 != bc1
        bc1 = fields.BoundaryCondition( name='periodic', indx=0, val=0 )
        assert bc0 == bc1

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

    def test_copy( self ):
        mesh = np.linspace( -0.05, 1.05, 12 )
        d = fields.Domain( mesh )
        f = fields.Field1D( 'x', d )

        data = np.asarray( range( len( f.val) ) )
        f.set_field( data )

        bcp = fields.BoundaryCondition( name='periodic' )
        f.add_boundary_condition( bcp )

        g = f.copy()

        assert g is not f
        assert g.name == f.name
        assert g.mesh == f.mesh
        assert np.all( g.val    == f.val    )
        assert np.all( g.bconds == f.bconds )

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

    def test__getitem__( self ):
        x = np.linspace( 0, 1, 11 )
        x = fields.Domain( x )
        f = fields.Field1D( 'q', x )

        data = np.asarray( range( len( x.xp ) ) )
        f.set_field( data )

        assert( np.all( f[:] == data ) )

        return

    def test__eq__( self ):
        a = np.linspace( 0, 1, 11 )
        x = fields.Domain( a )
        y = fields.Domain( a )
        f0 = fields.Field1D( 'q0', x )
        f1 = fields.Field1D( 'q1', x )
        f2 = fields.Field1D( 'q2', y )

        assert( f0 == f0 )
        assert( f0 != 0  )
        assert( f0 == f1 )
        assert( f0 != f2 )

        data = np.asarray( range( len( x.xp ) ) )
        f0.set_field( data )
        assert( f0 != f1 )
        f1.set_field( data )
        assert( f0 == f1 )

        bcp = fields.BoundaryCondition( name='periodic' )
        f0.add_boundary_condition( bcp )
        assert( f0 != f1 )
        f1.add_boundary_condition( bcp )
        assert( f0 == f1 )

        return

    def test__add__( self ):
        x = np.linspace( 0, 1, 11 )
        x = fields.Domain( x )
        f1 = fields.Field1D( 'q1', x )
        f2 = fields.Field1D( 'q2', x )
        f3 = fields.Field1D( 'q3', x )

        data = np.asarray( range( len( x.xp ) ) )
        f1.set_field(   data )
        f2.set_field( 2*data )
        f3.set_field( 3*data )

        assert( f1+f2 == f3 )
        return

    def test__sub__( self ):
        x = np.linspace( 0, 1, 11 )
        x = fields.Domain( x )
        f1 = fields.Field1D( 'q1', x )
        f4 = fields.Field1D( 'q4', x )
        f3 = fields.Field1D( 'q3', x )

        data = np.asarray( range( len( x.xp ) ) )
        f1.set_field(   data )
        f4.set_field( 4*data )
        f3.set_field( 3*data )

        assert( f4-f1 == f3 )
        return

    def test__mul__( self ):
        x = np.linspace( 0, 1, 11 )
        x = fields.Domain( x )
        f2 = fields.Field1D( 'q2', x )
        f3 = fields.Field1D( 'q3', x )
        f6 = fields.Field1D( 'q6', x )

        three = 3*np.ones( len( x.xp ) )

        data = np.asarray( range( len( x.xp ) ) )
        f2.set_field( 2*data )
        f3.set_field( three  )
        f6.set_field( 6*data )

        assert( f2*f3 == f6 )
        assert( f2*3  == f6 )
        return

    def test__div__( self ):
        x = np.linspace( 0, 1, 11 )
        x = fields.Domain( x )
        f2 = fields.Field1D( 'q2', x )
        f3 = fields.Field1D( 'q6', x )
        f6 = fields.Field1D( 'q3', x )

        three = 3*np.ones( len( x.xp ) )

        data = np.asarray( range( len( x.xp ) ) )
        f2.set_field( 2*data )
        f3.set_field( three  )
        f6.set_field( 6*data )

        assert( f6/f3 == f2 )
        return

