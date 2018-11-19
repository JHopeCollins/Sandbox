import numpy as np
import matplotlib.pyplot as plt
import polys

jac = polys.OrthogonalJacobi()

Np = 10
Nq =  8

alpha, beta = 0, 0

x = np.linspace( -1, 1, 101 )

z  = polys.Legendre.zeros[  Nq]
w  = polys.Legendre.weights[Nq]
pw = jac.w( alpha, beta, z )

p00x = np.zeros( [ Np+1, 101 ] )
p00z = np.zeros( [ Np+1,  Nq ] )

inner = np.zeros( Np+1 )

for i in range( Np+1 ):
    p00x[i,:] = jac.P( alpha, beta, i, x )
    p00z[i,:] = jac.P( alpha, beta, i, z )

    inner[i] = 0.0
    for j in range( Nq ):
        inner[i] += p00z[i,j]*p00z[i,j]*w[j]*pw[j]

fig1, ax1 = plt.subplots( 1, 1 )

for i in range( Np ):
    ax1.plot( x, p00x[i,:] )

fig2, ax2 = plt.subplots( 1, 1 )

for i in range( 6 ):
    ax2.plot( x, p00x[i,:] )

