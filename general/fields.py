"""
Written by: J Hope-Collins ( joshua.hope-collins@eng.ox.ac.uk )

Abstract Domain class for 1D scalar fields with unspecified numerical scheme
"""

import numpy as np

class Domain( object ):
    def __init__( self, xh ):
        """
        initialise Domain with cell vertices, cell sizes and number of cells
        """
        self.xh  = xh.copy()
        self.dxh = np.diff( self.xh )
        self.nh  = len( self.dxh )
        return
