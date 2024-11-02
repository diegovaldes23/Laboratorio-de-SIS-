#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <fitsio.h>
#include <omp.h>
#include <math.h>

#define VALOR_BORDE 255 // Valor para identificar los píxeles de borde

// Estructura para almacenar un punto de borde
typedef struct {
    int x;
    int y;
} PuntoBorde;

// Estructura para almacenar una elipse
typedef struct {
    double ox, oy, alpha, beta, theta;
} Elipse;

// Función para leer la imagen FITS y extraer los datos en una matriz
void leer_fits(const char *filename, int **imagen, long *naxes) {
    fitsfile *fptr;   // Apuntador al archivo FITS
    int status = 0;   // Estado de error
    int anynul = 0;   // Para manejar valores nulos
    long fpixel = 1;  // Primer píxel a leer (comienza en 1)

    // Abrir el archivo FITS
    if (fits_open_file(&fptr, filename, READONLY, &status)) {
        fits_report_error(stderr, status); // Reportar errores
        exit(EXIT_FAILURE);
    }

    // Obtener las dimensiones de la imagen
    fits_get_img_size(fptr, 2, naxes, &status); // Leer dimensiones de la imagen (asumimos 2D)

    // Asignar memoria para la imagen
    *imagen = (int *)malloc(naxes[0] * naxes[1] * sizeof(int));

    // Leer los datos de la imagen
    fits_read_img(fptr, TINT, fpixel, naxes[0] * naxes[1], NULL, *imagen, &anynul, &status);
    
    if (status) {
        fits_report_error(stderr, status);
        exit(EXIT_FAILURE);
    }

    // Cerrar el archivo FITS
    fits_close_file(fptr, &status);
}

// Función para crear la lista de puntos de borde en paralelo
PuntoBorde* detectar_bordes(int *imagen, long *naxes, int *num_puntos_borde) {
    long width = naxes[0];
    long height = naxes[1];
    int *temp_bordes = (int *)malloc(width * height * sizeof(int)); // Lista temporal
    int contador = 0;

    // Procesamiento en paralelo para detectar puntos de borde
    #pragma omp parallel for reduction(+:contador)
    for (long y = 0; y < height; y++) {
        for (long x = 0; x < width; x++) {
            int valor = imagen[y * width + x];

            // Si es un píxel de borde (valor 255)
            if (valor == VALOR_BORDE) {
                int idx = y * width + x;
                temp_bordes[idx] = 1; // Marcar como borde
                contador++;           // Incrementar el número de puntos de borde
            } else {
                temp_bordes[y * width + x] = 0;
            }
        }
    }

    // Crear una lista de puntos de borde con el tamaño exacto
    PuntoBorde *puntos_borde = (PuntoBorde *)malloc(contador * sizeof(PuntoBorde));
    int index = 0;

    // Llenar la lista de puntos de borde
    for (long y = 0; y < height; y++) {
        for (long x = 0; x < width; x++) {
            if (temp_bordes[y * width + x] == 1) {
                puntos_borde[index].x = x;
                puntos_borde[index].y = y;
                index++;
            }
        }
    }

    *num_puntos_borde = contador; // Devolver el número de puntos de borde
    free(temp_bordes);            // Liberar memoria temporal
    return puntos_borde;
}

// Función para detectar elipses usando el algoritmo de Hough
void detectar_elipses(PuntoBorde *puntos_borde, int num_puntos_borde, double alpha_min, int num_betas, int num_threads1, int num_threads2) {
    Elipse *elipses = NULL;
    int num_elipses = 0;

    double delta_beta = (double)num_betas / (2.0 * num_betas);
    int *vote = (int *)calloc(num_betas, sizeof(int));

    // Paralelismo anidado para la detección de elipses
    #pragma omp parallel for num_threads(num_threads1)
    for (int i = 0; i < num_puntos_borde; i++) {
        for (int j = i + 1; j < num_puntos_borde; j++) {
            PuntoBorde t = puntos_borde[i];
            PuntoBorde u = puntos_borde[j];

            double ox = (t.x + u.x) / 2.0;
            double oy = (t.y + u.y) / 2.0;
            double alpha = sqrt(pow(u.x - t.x, 2) + pow(u.y - t.y, 2)) / 2.0;
            double theta = atan2(u.y - t.y, u.x - t.x);

            if (alpha < alpha_min) continue;

            int max_beta_votos = 0;
            double mejor_beta = 0;

            #pragma omp parallel for num_threads(num_threads2) reduction(max: max_beta_votos)
            for (int k = 0; k < num_puntos_borde; k++) {
                if (k == i || k == j) continue;

                PuntoBorde p = puntos_borde[k];
                double delta = sqrt(pow(p.x - ox, 2) + pow(p.y - oy, 2));
                double gamma = sin(fabs(theta)) * (p.y - oy) + cos(fabs(theta)) * (p.x - ox);

                double denominador = alpha * alpha - gamma * gamma;
                if (denominador <= 0) continue;

                double numerador = alpha * alpha * delta * delta - alpha * alpha * gamma * gamma;
                double beta = sqrt(numerador / denominador);

                int beta_index = (int)(beta / delta_beta);
                if (beta_index >= 0 && beta_index < num_betas) {
                    #pragma omp atomic
                    vote[beta_index]++;
                }

                if (vote[beta_index] > max_beta_votos) {
                    max_beta_votos = vote[beta_index];
                    mejor_beta = beta;
                }
            }

            if (mejor_beta > 0) {
                #pragma omp critical
                {
                    elipses = realloc(elipses, (num_elipses + 1) * sizeof(Elipse));
                    elipses[num_elipses].ox = ox;
                    elipses[num_elipses].oy = oy;
                    elipses[num_elipses].alpha = alpha;
                    elipses[num_elipses].beta = mejor_beta;
                    elipses[num_elipses].theta = theta;
                    num_elipses++;
                }
            }
        }
    }

    // Imprimir las elipses encontradas
    for (int i = 0; i < num_elipses; i++) {
        printf("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n",
               elipses[i].ox, elipses[i].oy, elipses[i].alpha, elipses[i].beta, elipses[i].theta);
    }

    free(vote);
    free(elipses);
}

int main(int argc, char *argv[]) {
    char *input_filename = NULL;
    double alpha_min = 0;
    double relative_vote = 0;
    int num_threads1 = 1, num_threads2 = 1;
    int num_betas = 100;

    int opt;
    while ((opt = getopt(argc, argv, "i:a:r:b:u:d:")) != -1) {
        switch (opt) {
            case 'i': input_filename = optarg; break;
            case 'a': alpha_min = atof(optarg); break;
            case 'r': relative_vote = atof(optarg); break;
            case 'b': num_betas = atoi(optarg); break;
            case 'u': num_threads1 = atoi(optarg); break;
            case 'd': num_threads2 = atoi(optarg); break;
            default:
                fprintf(stderr, "Uso: %s -i imagen.fits -a alpha_min -r relative_vote -b num_betas -u num_threads1 -d num_threads2\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (!input_filename || !alpha_min || !relative_vote || !num_threads1 || !num_threads2) {
        fprintf(stderr, "Argumentos inválidos.\n");
        return EXIT_FAILURE;
    }

    omp_set_num_threads(num_threads1);

    int *imagen = NULL;
    long naxes[2];
    leer_fits(input_filename, &imagen, naxes);

    int num_puntos_borde = 0;
    PuntoBorde *puntos_borde = detectar_bordes(imagen, naxes, &num_puntos_borde);

    double start_time = omp_get_wtime();
    detectar_elipses(puntos_borde, num_puntos_borde, alpha_min, num_betas, num_threads1, num_threads2);
    double end_time = omp_get_wtime();

    printf("Tiempo total: %f segundos\n", end_time - start_time);

    free(imagen);
    free(puntos_borde);

    return 0;
}
