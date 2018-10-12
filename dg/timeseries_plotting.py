"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

plotting discontinuous galerkin timeseries data in a scrollable plot
"""

import numpy              as np
import matplotlib.pyplot  as plt

def previous_slice( ax ):
    f = ax.field
    ax.idx = ( ax.idx -1 ) % f.history.shape[0]
    #ax.lines[0].set_ydata( f.history[ax.idx,:] )

    qL = f.mesh.Lm1.reshape( f.mesh.nh, f.p ) * f.history[ax.idx,:].reshape( f.mesh.nh, f.p )
    qR = f.mesh.Lp1.reshape( f.mesh.nh, f.p ) * f.history[ax.idx,:].reshape( f.mesh.nh, f.p )

    qL = np.sum( qL, axis=1 )
    qR = np.sum( qR, axis=1 )

    for k in range( f.mesh.nh ):
        kL = k*f.p
        kR = kL + f.p

        q = f.history[ax.idx, kL:kR]
        q = np.append( q, qR[k] )
        q = np.append( qL[k], q )

        ax.lines[k].set_ydata( q )

def next_slice( ax ):
    f = ax.field
    ax.idx = ( ax.idx +1 ) % f.history.shape[0]
    #ax.lines[0].set_ydata( f.history[ax.idx,:] )

    qL = f.mesh.Lm1.reshape( f.mesh.nh, f.p ) * f.history[ax.idx,:].reshape( f.mesh.nh, f.p )
    qR = f.mesh.Lp1.reshape( f.mesh.nh, f.p ) * f.history[ax.idx,:].reshape( f.mesh.nh, f.p )

    qL = np.sum( qL, axis=1 )
    qR = np.sum( qR, axis=1 )

    for k in range( f.mesh.nh ):
        kL = k*f.p
        kR = kL + f.p

        q = f.history[ax.idx, kL:kR]
        q = np.append( q, qR[k] )
        q = np.append( qL[k], q )

        ax.lines[k].set_ydata( q )

def process_key( event ):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'up':
        previous_slice( ax )
    elif event.key == 'down':
        next_slice( ax )
    fig.canvas.draw()

def process_scroll( event ):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.button == 'down':
        previous_slice( ax )
    elif event.button == 'up':
        next_slice( ax )
    fig.canvas.draw()

def view_timeseries1D( f ):
    fig, ax = plt.subplots()
    ax.idx = 0
    ax.field = f

    qL = f.mesh.Lm1.reshape( f.mesh.nh, f.p ) * f.history[0,:].reshape( f.mesh.nh, f.p )
    qR = f.mesh.Lp1.reshape( f.mesh.nh, f.p ) * f.history[0,:].reshape( f.mesh.nh, f.p )

    qL = np.sum( qL, axis=1 )
    qR = np.sum( qR, axis=1 )

    for k in range( f.mesh.nh ):
        kL = k*f.p
        kR = kL + f.p

        x = f.mesh.xp[kL:kR]
        x = np.append( x, f.mesh.xh[k+1] )
        x = np.append( f.mesh.xh[k], x   )

        q = f.history[0, kL:kR]
        q = np.append( q, qR[k] )
        q = np.append( qL[k], q )

        ax.plot( x, q )

    fig.show()
    fig.canvas.mpl_connect(    'scroll_event', process_scroll )
    fig.canvas.mpl_connect( 'key_press_event', process_key    )
    return


