import matplotlib
matplotlib.use('TkAgg')  # Agrega esta línea al inicio

import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.patches import Ellipse

o_x = float(sys.argv[1])
o_y = float(sys.argv[2])
major = 2*float(sys.argv[3])
minor = 2*float(sys.argv[4])
theta = float(sys.argv[5])

fig, ax = plt.subplots()
ax.set(xlim=(0, 255), ylim=(255, 0), aspect='equal')
ellipse = Ellipse((o_x, o_y), major, minor, angle=theta, alpha=1)
ax.add_artist(ellipse)
ellipse.set_edgecolor('k')
ellipse.set_facecolor('w')

plt.show()  # O reemplaza con plt.savefig("ellipse.png")

plt.savefig("ellipse.png")