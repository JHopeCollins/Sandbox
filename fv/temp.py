import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots(1,1)

ax1.plot( x.xp[lb:ub], q[0][lb:ub], label=r'$\rho$'  )
ax1.plot( x.xp[lb:ub], q[1][lb:ub], label=r'$\rho U$' )
ax1.plot( x.xp[lb:ub], q[2][lb:ub], label=r'$\rho E$' )

ax1.legend()

fig1.show()

