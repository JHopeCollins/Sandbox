"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Test suite for fv.fluxclass.py file
Includes tests for Flux1D class
"""

import numpy as np
from sandbox import fv
from sandbox import general

class Test_Flux1D( object ):
    def test_init( self ):
        f = fv.fluxclass.Flux1D()
        assert f.stencil_radius == 1
        return

    def test_arg_list( self ):
        mesh = np.linspace( 0.0, 1.0, 11 )
        x = fv.fields.Domain( mesh )
        q = fv.fields.Field1D( 'q', x )

        f = fv.fluxclass.Flux1D()

        args = f.arg_list( q )
        assert len(args) == 3
        assert q.val is args[0]
        assert x.dxp is args[1]
        assert x.dxh is args[2]

        return

    def test_flux_calculation( self ):
        mesh = np.linspace( -0.05, 1.05, 12 )
        x = fv.fields.Domain( mesh )
        q = fv.fields.Field1D( 'q', x )

        f = fv.fluxclass.Flux1D()

        flux = f.flux_calculation( f.arg_list( q ) )

        assert len( flux ) == len( q.val ) + 1 - 2*f.stencil_radius
        assert np.all( flux == np.asarray( range( len( q.val ) +1 - 2*f.stencil_radius ) ) +1 )

        return

    def test_periodic( self ):
        mesh = np.linspace( -0.05, 1.05, 12 )
        x = fv.fields.Domain( mesh )
        q = fv.fields.Field1D( 'q', x )

        bcp = general.fields.BoundaryCondition( 'periodic' )
        q.add_boundary_condition( bcp )

        f = fv.fluxclass.Flux1D()
        r = f.stencil_radius

        flux = np.zeros( len( q.val ) +1 )

        f.periodic( bcp, q, flux )

        assert len( flux ) == len( q.val ) +1
        assert np.all( flux[r:-r] == 0 )
        assert flux[ 0] == flux[-1]
        assert flux[ 0] == 1

        return

    def test_apply( self ):
        mesh = np.linspace( -0.05, 1.05, 12 )
        x = fv.fields.Domain( mesh )
        q = fv.fields.Field1D( 'q', x )

        f = fv.fluxclass.Flux1D()
        r = f.stencil_radius

        fmiddle = f.flux_calculation( f.arg_list( q ) )
        flux    = f.apply( q )

        assert len( flux ) == len( q.val ) + 1
        assert np.all( flux[r:-r] == fmiddle )
        assert flux[ 0] == 0
        assert flux[-1] == 0

        bcp = general.fields.BoundaryCondition( 'periodic' )
        q.add_boundary_condition( bcp )

        flux = f.apply( q )
        assert len( flux ) == len( q.val ) + 1
        assert np.all( flux[r:-r] == fmiddle )
        assert flux[ 0] == flux[-1]
        assert flux[ 0] == 1

        return
