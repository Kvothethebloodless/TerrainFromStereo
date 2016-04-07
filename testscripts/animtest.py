import time
import numpy
import matplotlib.pyplot as plt


fig = plt.figure( 1 )
ax = fig.add_subplot( 111 )
ax.set_title("My Title")

im = ax.imshow( numpy.zeros( ( 256, 256, 3 ) ) ) # Blank starting image
fig.show()
im.axes.figure.canvas.draw()

tstart = time.time()
for a in xrange( 100 ):
  data = numpy.random.random( ( 256, 256, 3 ) ) # Random image to display
  ax.set_title( str( a ) )
  im.set_data( data )
  im.axes.figure.canvas.draw()

print ( 'FPS:', 100 / ( time.time() - tstart ) )