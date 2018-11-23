"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Test suite for fields.py file
Includes tests for Domain, Boundary Condition and Field1D classes
"""

import numpy as np
import fields
from sandbox import general


class Test_Domain( object ):
    def test_init( self ):
        mesh = np.linspace( 0, 1.0, 11 )
        d =         fields.Domain( mesh )
        g = general.fields.Domain( mesh )

        assert np.all( d.xp == 0.5*( g.xh[:-1] + g.xh[1:] ) )

        assert np.all( d.dxp == np.diff( d.xp ) )

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
        assert np.all( f.history == np.zeros( (1, len( f.val ) ) ) )

        return

    def test_set_field( self ):
        mesh = np.linspace( -0.05, 1.05, 12 )
        d = fields.Domain( mesh )
        f = fields.UnsteadyField1D( 'x', d )

        data = np.asarray( range( len( f.val ) ) )
        f.set_field( data )

        assert np.all( f.history[:] == f.val[:] )

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
        data = np.asarray( range( len( f.val ) ) )

        f.set_field( data )
        f.set_timestep( 0.1 )

        update = data[:]
        f.update( update )

        assert np.all( f.history[-1,:] == f.val[:] )
        assert f.t == 0.1

        return


class Test_VectorField1D( object ):
    def test_init( self ):
        x = np.linspace( 0, 1, 11 )
        x = fields.Domain( x )

        name = [ 'q', 'q0', 'q1' ]

        q = fields.VectorField1D( 2, name, x )

        assert q.n     == 2
        assert q.mesh  is x
        assert q.name  == 'q'
        assert q.names == name

        assert len( q.q ) == 2
        assert q.val.shape == ( 2, len( x.xp ) )
        for i, qi in enumerate( q.q ):
            assert qi.name == name[i+1]
            assert qi.mesh is x
            assert len( qi.val ) == len( x.xp )

        return

    def test_set_field( self ):
        x = np.linspace( 0, 1, 11 )
        x = fields.Domain( x )

        name = [ 'q', 'q0', 'q1' ]

        q = fields.VectorField1D( 2, name, x )

        assert np.all( q.val == 0 )
        data = np.ones( q.val.shape )
        data[1,:] *= 2
        q.set_field( data )

        assert np.all( q.val == data )
        assert np.all( q.q[0][:] == data[0,:] )
        assert np.all( q.q[1][:] == data[1,:] )

        return

    def test_copy( self ):
        x = np.linspace( 0, 1, 11 )
        x = fields.Domain( x )

        name = [ 'q', 'q0', 'q1' ]

        q = fields.VectorField1D( 2, name, x )

        bc0 = general.fields.BoundaryCondition(name='periodic' )
        bc1 = general.fields.BoundaryCondition(name='periodic' )

        q.add_boundary_condition( 0, bc0 )
        q.add_boundary_condition( 1, bc1 )

        q2 = q.copy()
        assert q2 is not q
        assert q2.n    == q.n
        assert q2.name == q.name
        assert q2.mesh == q.mesh

        for i in range( q2.n ):
            assert q2.q[i] is not q.q[i]
            assert q2.q[i] ==     q.q[i]

        return

    def test_add_boundary_condition( self ):
        x = np.linspace( 0, 1, 11 )
        x = fields.Domain( x )

        name = [ 'q', 'q0', 'q1' ]

        q = fields.VectorField1D( 2, name, x )

        bc0 = general.fields.BoundaryCondition(name='periodic' )
        bc1 = general.fields.BoundaryCondition(name='periodic' )

        q.add_boundary_condition(   0,  bc0 )
        assert bc0 in q.q[0].bconds
        q.add_boundary_condition( 'q1', bc1 )
        assert bc1 in q.q[1].bconds
        pass

    def test_update_ghosts( self ):
        pass

    def test_periodic( self ):
        pass

    def test__getitem__( self ):
        x = np.linspace( 0, 1, 11 )
        x = fields.Domain( x )

        name = [ 'q', 'q0', 'q1' ]

        q = fields.VectorField1D( 2, name, x )

        assert q[0] is q.q[0]
        assert q[1] is q.q[1]
        return

    def test__add__( self ):
        x = np.linspace( 0, 1, 11 )
        x = fields.Domain( x )

        name = [ 'q', 'q0', 'q1' ]

        q1 = fields.VectorField1D( 2, name, x )
        q2 = fields.VectorField1D( 2, name, x )

        data1 = np.ones( q1.val.shape )
        data1[1,:] *= 2
        q1.set_field( data1 )

        data2 = np.ones( q2.val.shape )
        data2[0,:] *= 3
        data2[1,:] *= 4
        q2.set_field( data2 )

        q3 = q1 + q2

        assert q3 is not q1
        assert q3 is not q2

        assert np.all( q3.val == q1.val + q2.val )

        return

    def test__sub__( self ):
        x = np.linspace( 0, 1, 11 )
        x = fields.Domain( x )

        name = [ 'q', 'q0', 'q1' ]

        q1 = fields.VectorField1D( 2, name, x )
        q2 = fields.VectorField1D( 2, name, x )

        data1 = np.ones( q1.val.shape )
        data1[1,:] *= 2
        q1.set_field( data1 )

        data2 = np.ones( q2.val.shape )
        data2[0,:] *= 3
        data2[1,:] *= 4
        q2.set_field( data2 )

        q3 = q1 - q2

        assert q3 is not q1
        assert q3 is not q2

        assert np.all( q3.val == q1.val - q2.val )

        return

    def test__mul__( self ):
        x = np.linspace( 0, 1, 11 )
        x = fields.Domain( x )

        name = [ 'q', 'q0', 'q1' ]

        q1 = fields.VectorField1D( 2, name, x )
        q2 = fields.VectorField1D( 2, name, x )

        data1 = np.ones( q1.val.shape )
        data1[1,:] *= 2
        q1.set_field( data1 )

        data2 = np.ones( q2.val.shape )
        data2[0,:] *= 3
        data2[1,:] *= 4
        q2.set_field( data2 )

        q3 = q1 * q2
        q4 = q1*2

        assert q3 is not q1
        assert q3 is not q2

        assert np.all( q3.val == q1.val * q2.val )
        assert np.all( q4.val == 2*q1.val )

        return

    def test__div__( self ):
        x = np.linspace( 0, 1, 11 )
        x = fields.Domain( x )

        name = [ 'q', 'q0', 'q1' ]

        q1 = fields.VectorField1D( 2, name, x )
        q2 = fields.VectorField1D( 2, name, x )

        data1 = np.ones( q1.val.shape )
        data1[1,:] *= 2
        q1.set_field( data1 )

        data2 = np.ones( q2.val.shape )
        data2[0,:] *= 3
        data2[1,:] *= 4
        q2.set_field( data2 )

        q3 = q1 / q2
        q4 = q1/2

        assert q3 is not q1
        assert q3 is not q2

        assert np.all( q3.val == q1.val / q2.val )
        assert np.all( q4.val == q1.val/2 )

        return

    def test__eq__( self ):
        x = np.linspace( 0, 1, 11 )
        y = fields.Domain( x )
        x = fields.Domain( x )

        name = [ 'q', 'q0', 'q1' ]

        q0 = fields.VectorField1D( 2, name, x )
        q1 = fields.VectorField1D( 2, name, x )
        q2 = fields.VectorField1D( 2, name, y )

        assert( q0 == q0 )
        assert( q0 !=  0 )
        assert( q0 == q1 )
        assert( q0 != q2 )

        q2 = q0.copy()

        data1 = np.ones( q1.val.shape )
        data1[1,:] *= 2
        q0.set_field( data1 )
        q1.set_field( data1 )

        data2 = np.ones( q2.val.shape )
        data2[0,:] *= 3
        data2[1,:] *= 4
        q2.set_field( data2 )

        assert( q0 == q1 )
        assert( q0 != q2 )

        q2 = q0.copy()

        bc0 = general.fields.BoundaryCondition( name='periodic' )
        bc1 = general.fields.BoundaryCondition( name='neumann', indx=0, val=0 )

        assert( bc0 != bc1 )

        q0.add_boundary_condition( 0, bc0 )
        q1.add_boundary_condition( 0, bc0 )
        q2.add_boundary_condition( 0, bc1 )

        assert( q0 == q1 )
        assert( q0 != q2 )

        q2 = q0.copy()
        q2.add_boundary_condition( 1, bc0 )
        assert( q0 != q2 )

        return
