"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Test suite for advective_fluxschemes.py file
Includes tests for AdvectiveFlux class
"""

import numpy as np
from sandbox import fv
import advective_fluxclasses as afx
import reconstructions as rc
import evolutions as ev

class Test_AdvectiveFlux1D( object ):
    def test_set_advection_velocity( self ):
        mesh = np.linspace( 0, 1, 11 )
        x = fv.fields.Domain( mesh )

        c = fv.fields.Field1D( 'c', x )
        c.set_field( np.ones( 10 ) )

        f = afx.AdvectiveFlux1D()
        f.set_advection_velocity( c )

        assert f.vel is c
        return

    def test_arg_list( self ):
        mesh = np.linspace( 0, 1, 11 )
        x = fv.fields.Domain( mesh )

        c = fv.fields.Field1D( 'c', x )
        c.set_field( np.ones( 10 ) )

        q = fv.fields.Field1D( 'q', x )
        q.set_field( np.zeros( 10 ) )

        f = afx.AdvectiveFlux1D()
        f.set_advection_velocity( c )

        args = f.arg_list( q )

        assert len( args ) == 4
        assert args[0] is q.val
        assert args[1] is x.dxp
        assert args[2] is x.dxh
        assert args[3] is c.val

        return


class Test_UpwindFlux1D( object ):
    def test_cell_face_direction( self ):
        f = afx.UpwindFlux1D()

        # uface            -  +  -   -    +
        u = np.asarray( [-2, 1, 1, -2, -3, 5] )
        d = f.cell_face_direction( u )

        assert np.all( d == [-1, 1, -1, -1, 1] )
        return

    def test_cell_indxs( self ):
        f = afx.UpwindFlux1D()

        u = np.asarray( [0, -2, 1, 1, -2, -3, 5] )
        i = f.cell_indxs( u )

        assert np.all( i == [0, 1, 2, 3, 4, 5] )
        return

    def test_upwind_indx( self ):
        f = afx.UpwindFlux1D()

        u = np.asarray( [-2, 1, 1, -2, -3, 5] )
        i = f.cell_indxs( u )
        d = f.cell_face_direction( u )

        up1 = f.upwind_indx( i, d, 1 )
        up2 = f.upwind_indx( i, d, 2 )
        assert np.all( up1 == [1, 1, 3, 4, 4] )
        assert np.all( up2 == [2, 0, 4, 5, 3] )
        return

    def test_downwind_indx( self ):
        f = afx.UpwindFlux1D()

        u = np.asarray( [-2, 1, 1, -2, -3, 5] )
        i = f.cell_indxs( u )
        d = f.cell_face_direction( u )

        down1 = f.downwind_indx( i, d, 1 )
        down2 = f.downwind_indx( i, d, 2 )
        assert np.all( down1 == [ 0, 2, 2, 3, 5] )
        assert np.all( down2 == [-1, 3, 1, 2, 6] )
        return


class Test_REAFlux1D( object ):
    def test_set_reconstruction_radius( self ):
        f = afx.REAFlux1D()
        f.set_reconstruction_radius( 3 )

        assert f.stencil_radius == 3
        return

    def test_set_reconstruction( self ):
        f = afx.REAFlux1D()
        f.set_reconstruction( rc.PCM1 )

        assert f.reconstruct == rc.PCM1
        return

    def test_set_evolution( self ):
        f = afx.REAFlux1D()
        f.set_evolution( ev.upwind1 )

        assert f.evolve == ev.upwind1
        return

    def test_flux_calculation( self ):
        f = afx.REAFlux1D()
        f.set_reconstruction( rc.PCM1 )
        f.set_reconstruction_radius( 1 )
        f.set_evolution( ev.upwind1 )

        #                 g  1  2   3  g
        q  = np.asarray( [1, 2, 3,  4, 5] )
        v  = np.asarray( [1, 1, 1, -3, 1] )
        h  = np.asarray( [1, 1, 1,  1, 1] )
        dx = np.asarray( [1, 1, 1,  1] )

        flux = f.flux_calculation( [q, dx, h, v] )

        assert np.all( flux == [1, 2, -12, 5] )
        return

