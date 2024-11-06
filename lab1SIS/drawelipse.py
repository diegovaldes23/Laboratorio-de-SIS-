import matplotlib
matplotlib.use('TkAgg')  # Agrega esta línea al inicio

import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.patches import Ellipse
from astropy.io import fits

# Parámetros
o_x = float(sys.argv[1])
o_y = float(sys.argv[2])
major = 2 * float(sys.argv[3])
minor = 2 * float(sys.argv[4])
theta = float(sys.argv[5])

# Crear figura y elipse
fig, ax = plt.subplots()
ax.set(xlim=(0, 255), ylim=(255, 0), aspect='equal')
ellipse = Ellipse((o_x, o_y), major, minor, angle=theta, alpha=1)
ax.add_artist(ellipse)
ellipse.set_edgecolor('k')
ellipse.set_facecolor('w')

# Renderizar la figura en un arreglo de datos
fig.canvas.draw()
image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# Guardar el arreglo en formato FITS
hdu = fits.PrimaryHDU(image_data)
hdul = fits.HDUList([hdu])
hdul.writeto("ellipse.fits", overwrite=True)

# Cerrar la figura
plt.close(fig)
