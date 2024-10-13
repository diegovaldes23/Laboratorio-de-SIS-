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
void detectar_elipses(PuntoBorde *puntos_borde, int num_puntos_borde, double alpha_min) {
    // Crear una lista para almacenar las elipses encontradas
    Elipse *elipses = NULL;
    int num_elipses = 0;

    // Para cada par de puntos t y u, calculamos ox, oy, alpha, theta
    #pragma omp parallel for
    for (int i = 0; i < num_puntos_borde; i++) {
        for (int j = i + 1; j < num_puntos_borde; j++) {
            PuntoBorde t = puntos_borde[i];
            PuntoBorde u = puntos_borde[j];

            // Calcular el centro (ox, oy), el semieje mayor (alpha) y la inclinación (theta)
            double ox = (t.x + u.x) / 2.0;
            double oy = (t.y + u.y) / 2.0;
            double alpha = sqrt(pow(u.x - t.x, 2) + pow(u.y - t.y, 2)) / 2.0;
            double theta = atan2(u.y - t.y, u.x - t.x);

            // Filtrar elipses con α menor al valor mínimo establecido
            if (alpha < alpha_min) {
                continue;
            }

            // Crear un acumulador para los votos de beta
            int max_beta_votos = 0;
            double mejor_beta = 0;

            // Para cada otro punto k distinto de t y u, calcular beta
            for (int k = 0; k < num_puntos_borde; k++) {
                if (k == i || k == j) continue;

                PuntoBorde p = puntos_borde[k];

                // Calcular delta y gamma para el punto k
                double delta = sqrt(pow(p.x - ox, 2) + pow(p.y - oy, 2));
                double gamma = sin(fabs(theta)) * (p.y - oy) + cos(fabs(theta)) * (p.x - ox);

                // Evitar cálculos inestables (denominador cercano a cero)
                double denominador = alpha * alpha - gamma * gamma;
                if (denominador <= 0) {
                    continue; // Saltar este cálculo si el denominador es cero o negativo
                }

                // Calcular beta usando la fórmula
                double numerador = alpha * alpha * delta * delta - alpha * alpha * gamma * gamma;
                double beta = sqrt(numerador / denominador);

                // Filtrar valores de beta demasiado grandes o pequeños
                if (beta > 0 && beta < 100) { // Ajusta este rango según tu imagen
                    int votos = 1; // Aquí puedes implementar un sistema de votación más complejo

                    // Si este valor de beta tiene más votos, lo consideramos el mejor
                    if (votos > max_beta_votos) {
                        max_beta_votos = votos;
                        mejor_beta = beta;
                    }
                }
            }

            // Si encontramos un buen valor para beta, guardamos la elipse
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
        printf("Elipse encontrada: (ox = %.2f, oy = %.2f, α = %.2f, β = %.2f, θ = %.2f)\n",
               elipses[i].ox, elipses[i].oy, elipses[i].alpha, elipses[i].beta, elipses[i].theta);
    }

    // Liberar la memoria de las elipses
    free(elipses);
}

int main(int argc, char *argv[]) {
    char *input_filename = NULL;
    double alpha_min = 0;
    double relative_vote = 0;
    int num_threads = 1;

    // Manejar los argumentos de línea de comandos usando getopt()
    int opt;
    while ((opt = getopt(argc, argv, "i:a:r:T:")) != -1) {
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
            case 'T':
                num_threads = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Uso: %s -i imagen.fits -a alpha_min -r relative_vote -T num_threads\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (input_filename == NULL || alpha_min == 0 || relative_vote == 0 || num_threads == 0) {
        fprintf(stderr, "Argumentos inválidos. Uso: %s -i imagen.fits -a alpha_min -r relative_vote -T num_threads\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Configurar el número de hebras (threads) para OpenMP
    omp_set_num_threads(num_threads);

    // Leer la imagen FITS
    int *imagen = NULL;
    long naxes[2];
    leer_fits(input_filename, &imagen, naxes);

    // Detectar los píxeles de borde y crear la lista de puntos de borde
    int num_puntos_borde = 0;
    PuntoBorde *puntos_borde = detectar_bordes(imagen, naxes, &num_puntos_borde);

    // Detectar las elipses usando los puntos de borde
    detectar_elipses(puntos_borde, num_puntos_borde, alpha_min);

    // Liberar la memoria
    free(imagen);
    free(puntos_borde);

    return 0;
}
