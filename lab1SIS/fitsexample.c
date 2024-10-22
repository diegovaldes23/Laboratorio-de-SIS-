#include <stdio.h>
#include "fitsio.h"

int main(int argc, char *argv[]) {
    fitsfile *fptr;
    int status = 0;
    int naxis;
    long naxes[2] = {1, 1};
    double fpixel = 1;

    if (argc != 2) {
        printf("Uso: %s nombre_de_archivo_fits\n", argv[0]);
        return 1;
    }

    // Abrir archivo FITS
    if (fits_open_file(&fptr, argv[1], READONLY, &status)) {
        fits_report_error(stderr, status);
        return status;
    }

    // Obtener dimensiones de la imagen
    if (fits_get_img_dim(fptr, &naxis, &status) || fits_get_img_size(fptr, 2, naxes, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return status;
    }

    if (naxis != 2) {
        printf("Error: El archivo FITS no es una imagen 2D\n");
        fits_close_file(fptr, &status);
        return 1;
    }

    printf("Tamaño de la imagen: %ld x %ld\n", naxes[0], naxes[1]);

    // Reservar memoria para la imagen
    double *myimage = (double *) malloc(naxes[0] * naxes[1] * sizeof(double));
    if (myimage == NULL) {
        printf("Error al asignar memoria\n");
        fits_close_file(fptr, &status);
        return 1;
    }

    // Leer los datos de la imagen
    if (fits_read_img(fptr, TDOUBLE, fpixel, naxes[0] * naxes[1], NULL, myimage, NULL, &status)) {
        fits_report_error(stderr, status);
        free(myimage);
        fits_close_file(fptr, &status);
        return status;
    }

    // Procesar los datos de la imagen
    double max = -1.0e-6;
    for (long i = 0; i < naxes[0] * naxes[1]; i++) {
        if (myimage[i] > max)
            max = myimage[i];
    }

    printf("Máximo valor de píxel: %f\n", max);

    // Limpiar y cerrar archivo FITS
    free(myimage);
    fits_close_file(fptr, &status);
    return 0;
}
