"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

Integrator functions for ODEs of the form: dqdt + L(q) = 0
"""

def EulerForward1( dt, q, L ):
    """
    Return change in q over timestep dt according to explicit first order Euler forward scheme

    integrates ODE of form dqdt + L(q) = 0 forward in time by dt
    """
    return -L(q)*dt

def RungeKutta4( dt, q, L ):
    """
    Return change in q over timestep dt according to explicit fourth order Runge-Kutta scheme

    integrates ODE of form dqdt + L(q) = 0 forward in time by dt
    """

    dq1 = EulerForward1( dt, q, L )

    r = q.copy()
    r.update( 0.5*dq1 )

    dq2 = EulerForward1( dt, r, L )

    r = q.copy()
    r.update( 0.5*dq2 )

    dq3 = EulerForward1( dt, r, L )

    r = q.copy()
    r.update( dq3 )

    dq4 = EulerForward1( dt, r, L )

    return (dq1 + 2*dq2 + 2*dq3 + dq4)*0.166666666667

