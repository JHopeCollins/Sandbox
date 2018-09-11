
"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Finite volume advective flux classes for 1D constant spacing structured meshes
"""

import numpy as np
import maths_utils as mth
import advective_fluxclasses as afc


class CDS2( afc.AdvectiveFlux1D ):
    """
        Advective flux class
        Central Difference Scheme
        Second order
        Stencil: if f=f(q(x)) f_{i+1/2} = (f(q_{i}) + f(q_{i+1}))/2
        Calculate flux from reconstructed primitives

        Required keys in namelists:
        other_variable_names:
            Advection velocity
        parameter_names:
            None
    """

    def __init__( self ):
        super( self.__class__, self ).__init__()
        self.stencil_radius = 1
        return

    def flux_calculation( self, args ):
        """
            Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

            input arguments:
            var: variable to be transported
            other_var_list: [ advection velocity ]
            par_list: None

            returns:
            flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-1
        """
        var = args[0]
        dx  = args[1]
        h   = args[2]
        u   = args[3]

        wl = 0.5*h[1:-2]/dx[1:-1]
        wr = 1 - wl

        u_cellface   = ( wl*u[  1:-2] + wr*u[  2:-1] )
        var_cellface = ( wl*var[1:-2] + wr*var[2:-1] )

        flux = u_cellface*var_cellface

        return flux

    def dirichlet( self, bc, flux, args ):
        """
            Apply dirichlet boundary conditions.

            input arguments:
            bc_indx: index of boundary to apply condition to
            bc_val: value of var at boundary
            var: array of the variable being transported by the flux
            other_var_list: list of arrays of other flow variables needed for the flux calculation
            par_list: list of parameters needed for the flux calculation

            returns:
            fi_list: list of tuples. Each tuple contains a flux value (position 0 in tuple) and the index of the global flux array the value corresponds to (position 1 in tuple)
        """
        var = args[0]
        dx  = args[1]
        h   = args[2]
        u   = args[3]

        ghost = bc.indx
        first = mth.step_into_array( bc.indx, 1 )

        u_boundary   = ( u[ghost] + u[first] ) / 2.0
        var_boundary = bc.val

        flux[ bc.indx ] = u_boundary*var_boundary

        return


class CDS2_2( afc.AdvectiveFlux1D ):
    """
        Advective flux class
        Central Difference Scheme
        Second order
        Stencil: f_{i+1/2} = (f_{i} + f{i+1})/2
        Stencil applied to fluxes, not primitives
        Reconstruct flux from calculate flux distribution

        Required keys in namelists:
        other_variable_names:
            Advection velocity
        parameter_names:
            None
    """

    def __init__( self ):
        super( self.__class__, self ).__init__()
        self.stencil_radius = 1
        return

    def flux_calculation( self, args ):
        """
            Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

            input arguments:
            var: variable to be transported
            other_var_list: [ advection velocity ]
            par_list: None

            returns:
            flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-1
        """
        var = args[0]
        dx  = args[1]
        h   = args[2]
        u   = args[3]

        fl = u[1:-2]*var[1:-2]
        fr = u[2:-1]*var[2:-1]

        wl = 0.5*h[1:-2]/dx[1:-1]
        wr = 1 - wl

        flux = wl*fl + wr*fr

        return flux


class UDS1( afc.UpwindFlux1D ):
    """
        Advective flux class
        Upwind Difference Scheme
        First order
        Stencil: f_{i+1/2} = f_{i  }  if u_{i+1/2} > 0
                           = f_{i+1}  if u_{i+1/2} < 0

        Required keys in namelists:
        other_variable_names:
            Advection velocity
        parameter_names:
            None
    """

    def __init__( self ):
        super( self.__class__, self ).__init__()
        self.stencil_radius = 1
        return

    def flux_calculation( self, args ):
        """
            Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

            input arguments:
            var: variable to be transported
            other_var_list: [ advection velocity ]
            par_list: None

            returns:
            flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-1
        """
        var = args[0]
        dx  = args[1]
        h   = args[2]
        u   = args[3]

        direction = self.cell_face_direction( u )
        indxs     = self.cell_indxs( u )
        upwind    = self.upwind_indx( indxs, direction, 1 )

        flux = u[upwind]*var[upwind]

        return flux

    def dirichlet( self, bc, flux, args ):
        """
            Apply dirichlet boundary conditions.

            input arguments:
            bc_indx: index of boundary to apply condition to
            bc_val: value of var at boundary
            var: array of the variable being transported by the flux
            other_var_list: list of arrays of other flow variables needed for the flux calculation
            par_list: list of parameters needed for the flux calculation

            returns:
            fi_list: list of tuples. Each tuple contains a flux value (position 0 in tuple) and the index of the global flux array the value corresponds to (position 1 in tuple)
        """
        var = args[0]
        dx  = args[1]
        h   = args[2]
        u   = args[3]

        ghost = bc.indx
        first = mth.step_into_array( bc.indx, 1 )

        dn = bc.indx*2+1
        direction = np.sign( u[ghost] + u[first] ).astype( int )

        var_boundary = bc.val
        # if dn == direction:
        #     upwind = ghost
        # else:
        #     upwind = first
        upwind = ( dn*direction - 1 ) / ( -2 )
        upwind = mth.step_into_array( bc.indx, upwind )

        flux[ bc.index ] = u[upwind]*bc.val

        return


class QUICK3( afc.UpwindFlux1D ):
    """
        Advective flux class
        Quadratic Upwind Interpolation Scheme
        Third order
        Stencil: f_{i+1/2} = 3/4f_{i  } + 3/8f_{i+1} - 1/8f_{i-1}  if u_{i+1/2} > 0
                           = 3/4f_{i+1} + 3/8f_{i  } - 1/8f_{i+2}  if u_{i+1/2} < 0

        Required keys in namelists:
        other_variable_names:
            Advection velocity
        parameter_names:
            None
    """

    def __init__( self ):
        super( self.__class__, self ).__init__()
        self.stencil_radius = 2
        return

    def flux_calculation( self, args ):
        """
            Calculate the flux in the centre of the domain (cell faces where stencil does not overlap with boundary)

            input arguments:
            var: variable to be transported
            other_var_list: [ advection velocity ]
            par_list: None

            returns:
            flux: array of diffusive fluxes at the cell faces in the centre of the domain. len(flux)=len(var)-3
        """
        var = args[0]
        dx  = args[1]
        h   = args[2]
        u   = args[3]

        direction = self.cell_face_direction( u )
        indxs     = self.cell_indxs( u )

        upwind    = self.upwind_indx(   indxs, direction, 1 )
        upupwind  = self.upwind_indx(   indxs, direction, 2 )
        downwind  = self.downwind_indx( indxs, direction, 2 )

        #                      |<----h---->|
        # cell     |     D     |     U     |     UU     |
        # face                 e

        e_U  =                   0.5*h[  upwind]
        e_UU =                       h[  upwind] + 0.5*h[upupwind]
        D_e  = 0.5*h[downwind]
        D_U  = 0.5*h[downwind] + 0.5*h[  upwind]
        D_UU = 0.5*h[downwind] +     h[  upwind] + 0.5*h[upupwind]
        U_UU =                   0.5*h[  upwind] + 0.5*h[upupwind]

        w1 = ( e_U * e_UU ) / ( D_U  * D_UU )
        w2 = ( e_U * D_e  ) / ( U_UU * D_UU )

        f_u  = u[   upwind ]*var[   upwind ]
        f_uu = u[ upupwind ]*var[ upupwind ]
        f_d  = u[ downwind ]*var[ downwind ]

        flux = f_u + w1*( f_d - f_u ) + w2*( f_u - f_uu )

        return flux

