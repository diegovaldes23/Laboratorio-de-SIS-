import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Crear una imagen 2D de 100x100 píxeles llena de ceros (negro)
data = np.zeros((100, 100))

# Definir el centro de la circunferencia y su radio
centro_x, centro_y = 50, 50  # Centro de la imagen
radio = 30  # Radio de la circunferencia

# Dibujar la circunferencia en la imagen
for y in range(100):
    for x in range(100):
        distancia = np.sqrt((x - centro_x)**2 + (y - centro_y)**2)
        if abs(distancia - radio) < 1:  # Condición para que los píxeles formen un círculo delgado
            data[y, x] = 255  # Poner los píxeles del círculo en blanco

# Crear el archivo FITS a partir de la matriz de datos
hdu = fits.PrimaryHDU(data)
hdul = fits.HDUList([hdu])

# Guardar la imagen FITS en el disco
hdul.writeto('imagen.fits', overwrite=True)

print("Imagen FITS 'imagen.fits' creada correctamente.")

# ----------------------------
# Visualización de la imagen y las elipses
# ----------------------------

# Función para trazar una elipse
def plot_ellipse(ox, oy, alpha, beta, theta):
    t = np.linspace(0, 2*np.pi, 100)
    Ell = np.array([alpha * np.cos(t), beta * np.sin(t)])  
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  
    Ell_rot = np.dot(R, Ell)  
    plt.plot(ox + Ell_rot[0, :], oy + Ell_rot[1, :], 'r')  # Trazar la elipse en rojo

# Parámetros mejorados de las elipses detectadas
elipses_detectadas = [
    (54.00, 38.50, 27.68, 17.04, 0.35),
    (24.00, 39.00, 10.77, 1.17, 1.95),
    # Más elipses filtradas y mejoradas aquí
]

# Crear la figura
plt.figure()
plt.imshow(data, cmap='gray')  # Mostrar la imagen en escala de grises

# Dibujar las elipses detectadas
for elipse in elipses_detectadas:
    plot_ellipse(*elipse)

plt.xlim(0, 100)
plt.ylim(0, 100)
plt.gca().invert_yaxis()  # Invertir el eje y para que coincida con la visualización de imágenes
plt.gca().set_aspect('equal', adjustable='box')

# Guardar la imagen como PNG
plt.savefig('elipses_detectadas_mejoradas.png')
print("Imagen con las elipses detectadas mejoradas guardada como 'elipses_detectadas_mejoradas.png'.")
