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
