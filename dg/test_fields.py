"""
Written by: J Hope-Collins ( joshua.hope-collins@eng.ox.ac.uk )

Test suite for dg/fields.py file
"""

import numpy as np
import fields
import polys

class Test_Domain( object ):
    def test_init( self ):
        mesh = np.linspace( -1.0, 1.0, 11 )
        d = fields.Domain( xh=mesh )

        assert np.all( d.xh ==          mesh )
        assert np.all( d.dx == np.diff( mesh ) )
        assert d.nh == 10
        assert np.all( d.dg == 0.5*d.dx )

        return

    def test_set_expansion( self ):
        mesh = np.linspace( -1.0, 1.0, 11 )
        d = fields.Domain( xh=mesh )
        o = 3

        d.set_expansion( order=o )

        assert d.p == o

        assert np.all( d.zp == polys.Legendre.zeros[  o] )
        assert np.all( d.wp == polys.Legendre.weights[o] )

        assert len( d.xp ) == ( len( mesh ) -1 ) * o
        assert d.xp[-1] ==  1.0 - ( 1.0 - polys.Legendre.zeros[o][-1] )*0.1

        assert len( d.dg ) == len( mesh ) -1
        assert d.dg[2] == 0.5*d.dx[2]

        assert len( d.Lm1 ) == ( len( mesh ) -1 ) * o
        assert len( d.Lp1 ) == ( len( mesh ) -1 ) * o

        tm = [1.47883056, -0.666666667, 0.18783611]
        tp = [0.18783611, -0.666666667, 1.47883056]

        assert np.all( np.isclose( d.Lm1[:o], tm ) )
        assert np.all( np.isclose( d.Lp1[:o], tp ) )

        assert len( d.M ) == d.p*d.nh

        return

class Test_Field1D( object ):
    def test_init( self ):
        mesh = np.linspace( -1.0, 1.0, 11 )
        mesh = fields.Domain( xh=mesh )
        mesh.set_expansion( order=3 )

        f = fields.Field1D( 'q', mesh )

        assert f.name == 'q'
        assert f.mesh == mesh
        assert len( f.val ) == len( f.mesh.xp )
        assert np.all( f.val == 0 )
        assert f.bconds == []

        return

    def test_set_field( self ):
        mesh = np.linspace( -1.0, 1.0, 11 )
        mesh = fields.Domain( xh=mesh )
        mesh.set_expansion( order=3 )
        y = np.asarray( range( len( mesh.xp ) ) )

        f = fields.Field1D( 'q', mesh )
        f.set_field( y )

        assert np.all( f.val == y )
        return

    def test_copy( self ):
        mesh = np.linspace( -1.0, 1.0, 11 )
        mesh = fields.Domain( xh=mesh )
        mesh.set_expansion( order=3 )
        y = np.asarray( range( len( mesh.xp ) ) )

        f = fields.Field1D( 'q', mesh )
        f.set_field( y )

        g = f.copy()

        assert g is not f
        assert g.mesh is f.mesh
        assert g.name == f.name
        assert np.all( g.val    == f.val    )
        assert np.all( g.bconds == f.bconds )

        return

    def test_update( self ):
        mesh = np.linspace( -1.0, 1.0, 11 )
        mesh = fields.Domain( xh=mesh )
        mesh.set_expansion( order=3 )
        y = np.asarray( range( len( mesh.xp ) ) )

        f = fields.Field1D( 'q', mesh )
        f.set_field( y )

        f.update( y )

        assert np.all( f.val == 2*y )

        return


class Test_UnsteadyField1D( object ):
    def test_init( self ):
        mesh = np.linspace( -1.0, 1.0, 11 )
        mesh = fields.Domain( xh=mesh )
        mesh.set_expansion( order=3)

        y = np.asarray( range( len( mesh.xp ) ) )

        f = fields.UnsteadyField1D( 'q', mesh )

        assert f.dt == None
        assert f.nt == 0
        assert f.t  == 0
        assert f.save_interval == 1
        assert np.all( f.history == np.zeros( (1, len( f.val ) ) ) )

        return

    def test_set_field( self ):
        mesh = np.linspace( -1.0, 1.0, 11 )
        mesh = fields.Domain( xh=mesh )
        mesh.set_expansion( order=3)

        y = np.asarray( range( len( mesh.xp ) ) )

        f = fields.UnsteadyField1D( 'q', mesh )
        f.set_field( y )

        assert np.all( f.val == f.history[:] )

        return

    def test_set_timestep( self ):
        mesh = np.linspace( -1.0, 1.0, 11 )
        mesh = fields.Domain( xh=mesh )
        mesh.set_expansion( order=3)

        f = fields.UnsteadyField1D( 'q', mesh )
        f.set_timestep( 0.1 )

        assert f.dt == 0.1

        return

    def test_set_save_interval( self ):
        mesh = np.linspace( -1.0, 1.0, 11 )
        mesh = fields.Domain( xh=mesh )
        mesh.set_expansion( order=3)

        f = fields.UnsteadyField1D( 'q', mesh )
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
        mesh = np.linspace( -1.0, 1.0, 11 )
        mesh = fields.Domain( xh=mesh )
        mesh.set_expansion( order=3)

        y = np.asarray( range( len( mesh.xp ) ) )

        f = fields.UnsteadyField1D( 'q', mesh )
        f.set_field( y )
        f.set_timestep( 0.1 )

        f.update( y )

        assert np.all( f.history[-1,:] == f.val )
        assert f.t == 0.1

        return

