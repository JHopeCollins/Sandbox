"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Test suite for diffusive_fluxschemes.py file
Includes tests for DiffusiveFlux1D class
"""

import numpy as np
import fields
import diffusive_fluxschemes as dfx

class Test_DiffusiveFlux1D( object ):
    def test_set_diffusion_coefficient( self ):
        f = dfx.DiffusiveFlux1D()
        f.set_diffusion_coefficient( 0.5 )
        assert f.dcoeff == 0.5
        return

    def test_arg_list( self ):
        mesh = np.linspace( 0, 1, 11 )
        x = fields.Domain( mesh )

        f = dfx.DiffusiveFlux1D()
        f.set_mesh( x )
        f.set_diffusion_coefficient( 0.5 )

        q = fields.Field1D( 'q', x )

        args = f.arg_list ( q )

        assert len( args ) == 4
        assert args[0] is q.val
        assert args[1] is x.dx
        assert args[2] is x.h
        assert args[3] is f.dcoeff

        return
