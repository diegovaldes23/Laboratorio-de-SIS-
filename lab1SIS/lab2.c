#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <fitsio.h>
#include <math.h>
#include <omp.h>

#define VALOR_BORDE 255 // Valor para identificar los píxeles de borde

int num_threads_level1 = 1;
int num_threads_level2 = 1;

// Estructura para almacenar un punto de borde
typedef struct {
    int x;
    int y;
} PuntoBorde;

// Estructura para almacenar los parámetros de una elipse
typedef struct {
    double ox, oy, alpha, beta, theta;
} Elipse;

PuntoBorde* extract_borders(const char *filename, int *num_puntos_borde, long *naxes) {
    fitsfile *fptr; // Apuntador al archivo fits
    int status = 0; // Variable de estado para manejar errores
    int bitpix, naxis; // Tipo de datos de la imagen y número de dimensiones

    // Se abre el archivo fits en modo "lectura"
    if (fits_open_file(&fptr, filename, READONLY, &status)) {
        fits_report_error(stderr, status); // Reporta errores
        return NULL;
    }

    // Se obtiene el tipo de datos y dimensiones de la imagen
    if (fits_get_img_param(fptr, 2, &bitpix, &naxis, naxes, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return NULL;
    }

    // Se verifica que la imagen es de dos dimensiones
    if (naxis != 2) {
        fprintf(stderr, "Error: Se espera una imagen 2D.\n");
        fits_close_file(fptr, &status);
        return NULL;
    }

    // Se asigna memoria para la imagen
    long npixels = naxes[0] * naxes[1];
    int *imagen = (int *)malloc(npixels * sizeof(int));
    if (!imagen) {
        fprintf(stderr, "Error al asignar memoria para la imagen.\n");
        fits_close_file(fptr, &status);
        return NULL;
    }

    // Se leen los datos de la imagen en el arreglo
    long fpixel = 1; // Primer pixel a leer
    int anynul = 0; // Para manejar valores nulos
    if (fits_read_img(fptr, TINT, fpixel, npixels, NULL, imagen, &anynul, &status)) {
        fits_report_error(stderr, status);
        free(imagen);
        fits_close_file(fptr, &status);
        return NULL;
    }

    // Se cuentan y almacenan los puntos borde
    PuntoBorde *puntos_borde = malloc(npixels * sizeof(PuntoBorde));
    int contador = 0;
    for (long y = 0; y < naxes[1]; y++) {
        for (long x = 0; x < naxes[0]; x++) {
            if (imagen[y * naxes[0] + x] == VALOR_BORDE) {
                puntos_borde[contador].x = x;
                puntos_borde[contador].y = y;
                contador++;
            }
        }
    }
    // Se redimensiona el arreglo de puntos de borde para ajustarlo al tamaño exacto
    puntos_borde = realloc(puntos_borde, contador * sizeof(PuntoBorde));
    *num_puntos_borde = contador;

    // Se libera la memoria y se cierra el archivo fits
    free(imagen);
    fits_close_file(fptr, &status);
    return puntos_borde;
}

void detect_elipses(PuntoBorde *puntos_borde, int num_puntos_borde, double alpha_min, long *naxes, int num_betas, double relative_vote, Elipse *elipses, int *num_elipses) {
    int max_elipses = 1000;
    *num_elipses = 0;

    // Cálculo de delta_beta para la discretización de beta
    double delta_beta = ((double)naxes[0] / 2) / num_betas;

    // Se configura número de hilos de OpenMP en el nivel 1
    omp_set_num_threads(num_threads_level1);

    #pragma omp parallel for num_threads(num_threads_level1)
    for (int i = 0; i < num_puntos_borde; i++) {
        for (int j = i + 1; j < num_puntos_borde; j++) {
            PuntoBorde t = puntos_borde[i];
            PuntoBorde u = puntos_borde[j];

            // Cálculo del centro (ox, oy), semieje mayor (alpha) y ángulo (theta)
            double ox = (t.x + u.x) / 2.0;
            double oy = (t.y + u.y) / 2.0;
            double alpha = sqrt(pow(u.x - t.x, 2) + pow(u.y - t.y, 2)) / 2.0;
            double theta = atan2(u.y - t.y, u.x - t.x);

            if (alpha < alpha_min) continue;

            // Se inicializa el acumulador de votos para beta
            int vote[num_betas];
            memset(vote, 0, num_betas * sizeof(int));

            // Se itera sobre el resto de los puntos de borde para calcular beta
            #pragma omp parallel for num_threads(num_threads_level2)
            for (int k = 0; k < num_puntos_borde; k++) {
                if (k == i || k == j) continue;
                PuntoBorde p = puntos_borde[k];

                // Cálculo de delta y gamma
                double delta = sqrt(pow(p.y - oy, 2) + pow(p.x - ox, 2));
                double gamma = sin(theta) * (p.y - oy) + cos(theta) * (p.x - ox);

                // Se evitan divisiones por cero y se calcula beta
                double denominador = alpha * alpha - gamma * gamma;
                if (denominador <= 0) continue;

                double beta = sqrt(((alpha * alpha) * (delta * delta) - (alpha * alpha) * (gamma * gamma)) / denominador);

                // Se verifica que esté en un rango razonable
                if (beta < 0 || beta > naxes[0] / 2) continue;

                // Se discretiza beta
                int beta_index = (int)(beta / delta_beta);
                if (beta_index >= 0 && beta_index < num_betas) {
                    #pragma omp atomic
                    vote[beta_index]++;
                }
            }

            // Se almacenan las elipses encontradas
            for (int b = 0; b < num_betas; b++) {
                if (vote[b] >= relative_vote * num_puntos_borde) {
                    #pragma omp critical
                    {
                        if (*num_elipses >= max_elipses) {
                            max_elipses *= 2;
                            elipses = realloc(elipses, max_elipses * sizeof(Elipse));
                            if (!elipses) {
                                fprintf(stderr, "Error al reasignar memoria para elipses.\n");
                                exit(EXIT_FAILURE);
                            }
                        }
                        elipses[*num_elipses].ox = ox;
                        elipses[*num_elipses].oy = oy;
                        elipses[*num_elipses].alpha = alpha;
                        elipses[*num_elipses].beta = b * delta_beta;
                        elipses[*num_elipses].theta = theta;
                        (*num_elipses)++;
                    }
                }
            }
        }
    }
}

void imprimir_elipses(Elipse *elipses, int num_elipses) {
    for (int i = 0; i < num_elipses; i++) {
        printf("%.2f \t%.2f \t%.2f \t%.2f \t%.2f\n",
               elipses[i].ox, elipses[i].oy, elipses[i].alpha, elipses[i].beta, elipses[i].theta);
    }
}

int main(int argc, char *argv[]) {
    char *input_filename = NULL;
    double alpha_min = 0;
    double relative_vote = 0;
    int num_betas = 0;

    int opt;
    while ((opt = getopt(argc, argv, "i:a:r:b:u:d:")) != -1) {
        switch (opt) {
            case 'i':
                input_filename = optarg;
                break;
            case 'a':
                alpha_min = atof(optarg);
                break;
            case 'r':
                relative_vote = atof(optarg);
                break;
            case 'b':
                num_betas = atoi(optarg);
                break;
            case 'u':
                num_threads_level1 = atoi(optarg);
                break;
            case 'd':
                num_threads_level2 = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Uso: %s -i imagen.fits -a alpha_min -r relative_vote -b num_betas -u num_hebras1 -d num_hebras2\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (!input_filename || alpha_min == 0 || relative_vote == 0 || num_betas == 0 || num_threads_level1 == 0 || num_threads_level2 == 0) {
        fprintf(stderr, "Argumentos inválidos. Uso: %s -i imagen.fits -a alpha_min -r relative_vote -b num_betas -u num_hebras1 -d num_hebras2\n", argv[0]);
        return EXIT_FAILURE;
    }

    omp_set_nested(1); // Habilita el paralelismo anidado

    // 1. Tiempo de ejecución de todo el proceso con 1 hebra
    omp_set_num_threads(1);
    double start_time_serial = omp_get_wtime();

    long naxes[2];
    int num_puntos_borde = 0;
    PuntoBorde *puntos_borde = extract_borders(input_filename, &num_puntos_borde, naxes);

    if (puntos_borde) {
        int max_elipses = 1000;
        Elipse *elipses = malloc(max_elipses * sizeof(Elipse));
        int num_elipses = 0;

        detect_elipses(puntos_borde, num_puntos_borde, alpha_min, naxes, num_betas, relative_vote, elipses, &num_elipses);
        imprimir_elipses(elipses, num_elipses);

        free(puntos_borde);
        free(elipses);
    } else {
        fprintf(stderr, "Error al extraer puntos de borde.\n");
        return EXIT_FAILURE;
    }

    double end_time_serial = omp_get_wtime();
    double total_time_serial = end_time_serial - start_time_serial;

     // 2. Tiempo de ejecución de todo el proceso con el número de hebras especificado
    omp_set_num_threads(num_threads_level1);
    double start_time_parallel = omp_get_wtime();

    puntos_borde = extract_borders(input_filename, &num_puntos_borde, naxes);
    
    if(puntos_borde){
        int max_elipses = 1000;
        Elipse *elipses = malloc(max_elipses * sizeof(Elipse));
        int num_elipses = 0;
        detect_elipses(puntos_borde, num_puntos_borde, alpha_min, naxes, num_betas, relative_vote, elipses, &num_elipses);
        free(puntos_borde);
    } else {
        fprintf(stderr, "Error al extraer puntos de borde.\n");
        return EXIT_FAILURE;
    }
    double end_time_parallel = omp_get_wtime();
    double total_time_parallel = end_time_parallel - start_time_parallel;

    // 3. Porción serial del programa
    double porcion_serial = (total_time_serial - total_time_parallel) / total_time_serial;

    // 4. Speedup del programa
    double speedup_general = total_time_serial / total_time_parallel;

    // 5. Tiempo de ejecución solo de la transformada de Hough con 1 hebra
    omp_set_num_threads(1);
    puntos_borde = extract_borders(input_filename, &num_puntos_borde, naxes);
    double start_hough_serial = omp_get_wtime();
    int max_elipses = 1000;
    Elipse *elipses = malloc(max_elipses * sizeof(Elipse));
    int num_elipses = 0;
    detect_elipses(puntos_borde, num_puntos_borde, alpha_min, naxes, num_betas, relative_vote, elipses, &num_elipses);
    double end_hough_serial = omp_get_wtime();
    double time_hough_serial = end_hough_serial - start_hough_serial;
    free(puntos_borde);
    free(elipses);

    // 6. Tiempo de ejecución solo de la transformada de Hough con hebras especificadas
    omp_set_num_threads(num_threads_level1);
    puntos_borde = extract_borders(input_filename, &num_puntos_borde, naxes);
    double start_hough_parallel = omp_get_wtime();
    detect_elipses(puntos_borde, num_puntos_borde, alpha_min, naxes, num_betas, relative_vote, elipses, &num_elipses);
    double end_hough_parallel = omp_get_wtime();
    double time_hough_parallel = end_hough_parallel - start_hough_parallel;
    free(puntos_borde);
    free(elipses);

    // 7. Speedup de la transformada de Hough
    double speedup_hough = time_hough_serial / time_hough_parallel;

    // Impresión de resultados en el formato solicitado
    printf("%.2f\n%.2f\n%.2f\n%.2f\n%.2f\n", total_time_serial, total_time_parallel, porcion_serial * 100, speedup_general, time_hough_serial);
    printf("%.2f\n%.2f\n", time_hough_parallel, speedup_hough);

    return EXIT_SUCCESS;
}
