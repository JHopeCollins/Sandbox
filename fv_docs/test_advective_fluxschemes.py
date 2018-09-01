"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Test suite for advective_fluxschemes.py file
Includes tests for AdvectiveFlux class
"""

import numpy as np
import fields
import advective_fluxschemes as afx

class Test_AdvectiveFlux1D( object ):
    def test_set_advection_velocity( self ):
        mesh = np.linspace( 0, 1, 11 )
        x = fields.Domain( mesh )

        c = fields.Field1D( 'c', x )
        c.set_field( np.ones( 9 ) )

        f = afx.AdvectiveFlux1D()
        f.set_mesh( x )
        f.set_advection_velocity( c )

        assert f.vel is c
        return

    def test_arg_list( self ):
        mesh = np.linspace( 0, 1, 11 )
        x = fields.Domain( mesh )

        c = fields.Field1D( 'c', x )
        c.set_field( np.ones( 9 ) )

        q = fields.Field1D( 'q', x )
        q.set_field( np.zeros( 9 ) )

        f = afx.AdvectiveFlux1D()
        f.set_mesh( x )
        f.set_advection_velocity( c )

        args = f.arg_list( q )

        assert len( args ) == 3
        assert args[0] is q.val
        assert args[1] is x.dx
        assert args[2] is c.val

        return


