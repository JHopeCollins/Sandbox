"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Test suite for fields.py file
Includes tests for Domain, Boundary Condition and Field1D classes
"""

import numpy as np
import fields


class Test_Domain( object ):
    def test_init( self ):
        mesh = np.linspace( 0, 1.0, 11 )
        d = fields.Domain( mesh )

        assert np.all( d.x  ==          mesh   )
        assert np.all( d.dx == np.diff( mesh ) )

        assert np.all( d.x_noghost  == d.x[ 1:-1] )
        assert np.all( d.dx_noghost == d.dx[1:-1] )

        return

    def test_set_x( self ):
        mesh  = np.linspace(   0, 1.0, 11 )
        mesh2 = np.linspace( 1.0, 2.0, 11 )
        d = fields.Domain( mesh )
        d.set_x( mesh2 )

        assert np.all( d.x  ==          mesh2   )
        assert np.all( d.dx == np.diff( mesh2 ) )

        assert np.all( d.x_noghost  == d.x[ 1:-1] )
        assert np.all( d.dx_noghost == d.dx[1:-1] )

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
        mesh = np.linspace( 0, 1.0, 11 )
        d = fields.Domain( mesh )
        f = fields.Field1D( 'x', d )

        assert f.name == 'x'
        assert f.mesh == d
        assert len( f.val ) == len( f.mesh.x )
        assert np.all( f.val == 0 )
        assert np.all( f.val_noghost == f.val[1:-1] )
        assert f.bconds == []

        return

    def test_set_field( self ):
        mesh = np.linspace( -0.05, 1.05, 12 )
        d = fields.Domain( mesh )
        f = fields.Field1D( 'x', d )

        data = np.asarray( range( len( f.val_noghost ) ) )
        f.set_field( data )

        assert np.all( f.val_noghost == data )
        assert f.val[ 0] == 0
        assert f.val[-1] == 0

        return

    def test_copy( self ):
        mesh = np.linspace( -0.05, 1.05, 12 )
        d = fields.Domain( mesh )
        f = fields.Field1D( 'x', d )

        data = np.asarray( range( len( f.val_noghost ) ) )
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
        mesh = np.linspace( -0.05, 1.05, 12 )
        d = fields.Domain( mesh )
        f = fields.Field1D( 'x', d )

        data = np.asarray( range( len( f.val_noghost ) ) )
        f.set_field( data )

        update = data[:]

        f.update( update )

        assert np.all( f.val_noghost == 2*data )
        assert f.val[ 0] == 0
        assert f.val[-1] == 0

        return

    def test_periodic( self ):
        mesh = np.linspace( -0.05, 1.05, 12 )
        d = fields.Domain( mesh )
        f = fields.Field1D( 'x', d )

        data = np.asarray( range( len( f.val_noghost ) ) )
        f.set_field( data )

        bcp = fields.BoundaryCondition( name='periodic' )

        f.periodic( bcp )
        assert f.val[ 0] == f.val[-2]
        assert f.val[-1] == f.val[ 1]

        return

    def test_dirichlet( self ):
        mesh = np.linspace( -0.05, 1.05, 12 )
        d = fields.Domain( mesh )
        f = fields.Field1D( 'x', d )

        data = np.asarray( range( len( f.val_noghost ) ) )
        f.set_field( data )

        bcd0  = fields.BoundaryCondition( name='dirichlet', indx=0,  val=0 )
        bcdm1 = fields.BoundaryCondition( name='dirichlet', indx=-1, val=0 )

        f.dirichlet( bcd0  )
        f.dirichlet( bcdm1  )
        assert f.val[ 0] ==  0
        assert f.val[-1] == -9

        return

    def test_neumann( self ):
        mesh = np.linspace( -0.05, 1.05, 12 )
        d = fields.Domain( mesh )
        f = fields.Field1D( 'x', d )

        data = np.asarray( range( len( f.val_noghost ) ) )
        f.set_field( data )

        bcn0  = fields.BoundaryCondition( name='neumann',  indx=0,  val=1 )
        bcnm1 = fields.BoundaryCondition( name='neumann',  indx=-1, val=1 )

        f.neumann( bcn0  )
        f.neumann( bcnm1 )
        assert f.val[ 0] == -0.1
        assert f.val[-1] ==  9.1

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

    def test_update_ghosts( self ):
        mesh = np.linspace( -0.05, 1.05, 12 )
        d = fields.Domain( mesh )
        f = fields.Field1D( 'x', d )

        data = np.asarray( range( len( f.val_noghost ) ) ) + 1
        f.set_field( data )

        bcp = fields.BoundaryCondition( name='periodic' )
        f.bconds.append( bcp )
        f.update_ghosts(     )

        assert f.val[ 0] == f.val[-2]
        assert f.val[-1] == f.val[ 1]

        bcd0  = fields.BoundaryCondition( name='dirichlet', indx=0,  val=0 )
        bcdm1 = fields.BoundaryCondition( name='dirichlet', indx=-1, val=0 )
        f.add_boundary_condition( bcd0  )
        f.add_boundary_condition( bcdm1 )

        assert f.val[ 0] ==  -1
        assert f.val[-1] == -10

        return


class Test_UnsteadyField1D( object ):
    def test_init( self ):
        mesh = np.linspace( 0, 1.0, 11 )
        d = fields.Domain( mesh )
        f = fields.UnsteadyField1D( 'q', d )

        assert f.dt == None
        assert f.nt == 0
        assert f.t  == 0
        assert f.save_interval == 1
        assert np.all( f.history == np.zeros( (1, len( f.val_noghost ) ) ) )

        return

    def test_set_field( self ):
        mesh = np.linspace( -0.05, 1.05, 12 )
        d = fields.Domain( mesh )
        f = fields.UnsteadyField1D( 'x', d )

        data = np.asarray( range( len( f.val_noghost ) ) )
        f.set_field( data )

        assert np.all( f.history[:] == f.val_noghost[:] )

        return

    def test_set_timestep( self ):
        mesh = np.linspace( -0.05, 1.05, 12 )
        d = fields.Domain( mesh )
        f = fields.UnsteadyField1D( 'x', d )
        f.set_timestep( 0.1 )

        assert f.dt == 0.1

        return

    def test_set_save_interval( self ):
        mesh = np.linspace( -0.05, 1.05, 12 )
        d = fields.Domain( mesh )
        f = fields.UnsteadyField1D( 'x', d )
        f.set_timestep( 0.1 )

        f.set_save_interval()
        assert f.save_interval == 1

        f.set_save_interval( dt=0.1, nt=5 )
        assert f.save_interval == 1

        f.set_save_interval( dt=0.3 )
        assert f.save_interval == 3

        f.set_save_interval( nt=5 )
        assert f.save_interval == 5

        return

    def test_update( self ):
        mesh = np.linspace( -0.05, 1.05, 12 )

        d = fields.Domain( mesh )
        f = fields.UnsteadyField1D( 'x', d )
        data = np.asarray( range( len( f.val_noghost ) ) )

        f.set_field( data )
        f.set_timestep( 0.1 )

        update = data[:]
        f.update( update )

        assert np.all( f.history[-1,:] == f.val_noghost )
        assert f.t == 0.1

        return




