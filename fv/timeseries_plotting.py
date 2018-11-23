"""
Written by: J Hope-Collins (jth39@cam.ac.uk)

plotting finite volume timeseries data in a scrollable plot
"""

import numpy              as np
import matplotlib.pyplot  as plt

def previous_slice( ax ):
    f = ax.field
    ax.idx = ( ax.idx -1 ) % f.history.shape[0]
    ax.lines[0].set_ydata( f.history[ax.idx,:] )

def next_slice( ax ):
    f = ax.field
    ax.idx = ( ax.idx +1 ) % f.history.shape[0]
    ax.lines[0].set_ydata( f.history[ax.idx,:] )

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

def view_timeseries1D( field ):
    fig, ax = plt.subplots()
    ax.idx = 0
    ax.field = field
    lo = 0.95*np.min( field.history )
    hi = 1.05*np.max( field.history )
    ax.set_ylim( [ lo, hi ] )
    ax.plot( field.mesh.xp, field.history[0,:] )
    fig.show()

    fig.canvas.mpl_connect(    'scroll_event', process_scroll )
    fig.canvas.mpl_connect( 'key_press_event', process_key    )

