"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Test suite for reconstructions.py file
"""

import numpy as np
import reconstructions as r

def test_PCM1( ):
    u = np.asarray( [1, 2, 3, 4, 5] )

    uL, uR = r.PCM1( u, 1, 1 )

    assert np.all( uR == u[1:  ]  )
    assert np.all( uL == u[ :-1] )
    return
