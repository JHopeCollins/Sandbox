"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Test suite for equationclass.py file
Includes tests for Equation class
"""

import numpy as np
from sandbox import fv
import advective_fluxclasses as afx
import diffusive_fluxclasses as dfx
import equationclass
import ODEintegrators

class Test_Equation( object ):
    def test_init( self ):
        eq = equationclass.Equation()
        assert eq.flux_terms == []
        return

    def test_set_variable( self ):
        mesh = np.linspace( 0, 1, 11 )
        x = fv.fields.Domain( mesh )

        q = fv.fields.Field1D( 'q', x )
        q.set_field( np.ones( 10 ) )

        eq = equationclass.Equation()
        eq.set_variable( q )

        assert eq.q is q
        assert eq.mesh is q.mesh

        return

    def test_time_integration( self ):
        eq = equationclass.Equation()
        eq.set_time_integration( ODEintegrators.EulerForward1 )
        assert eq.time_stepper == ODEintegrators.EulerForward1
        return

    def test_add_flux_term( self ):
        fa = afx.AdvectiveFlux1D()
        fd = dfx.DiffusiveFlux1D()

        eq = equationclass.Equation()
        eq.add_flux_term( fa )

        assert len( eq.flux_terms ) == 1
        assert fa in eq.flux_terms

        eq.add_flux_term( fd )

        assert len( eq.flux_terms ) == 2
        assert fa in eq.flux_terms
        assert fd in eq.flux_terms

        return

    def test_spatial_operator( self ):
        mesh = np.linspace( 0, 1, 11 )
        x = fv.fields.Domain( mesh )

        c = fv.fields.Field1D( 'c', x )
        c.set_field( np.ones( 10 ) )

        q = fv.fields.Field1D( 'q', x )
        q.set_field( np.zeros( 10 ) )

        fa = afx.AdvectiveFlux1D()
        fa.set_advection_velocity( c )

        fd = dfx.DiffusiveFlux1D()
        fd.set_diffusion_coefficient( 0.5 )

        eq = equationclass.Equation()
        eq.add_flux_term( fa )
        eq.add_flux_term( fd )

        dq = eq.spatial_operator( q )

        dq_test = fa.apply( q ) + fd.apply( q )
        dq_test = np.diff( dq_test )/ x.dxh

        assert np.all( dq == dq_test )

        return

    def test_step( self ):
        mesh = np.linspace( 0, 1, 11 )
        x = fv.fields.Domain( mesh )

        c = fv.fields.Field1D( 'c', x )
        c.set_field( np.ones( 10 ) )

        q = fv.fields.Field1D( 'q', x )
        q.set_field( np.zeros( 10 ) )
        assert( type(q) == fv.fields.Field1D )

        fa = afx.AdvectiveFlux1D()
        fa.set_advection_velocity( c )

        fd = dfx.DiffusiveFlux1D()
        fd.set_diffusion_coefficient( 0.5 )

        eq = equationclass.Equation()
        eq.set_variable( q )
        eq.set_time_integration( ODEintegrators.EulerForward1 )
        eq.add_flux_term( fa )
        eq.add_flux_term( fd )

        r = q.copy()

        dr = ODEintegrators.EulerForward1( 0.1, r, eq.spatial_operator )
        r.update( dr )

        eq.step( 0.1 )

        assert np.all( q.val == r.val )

        return


