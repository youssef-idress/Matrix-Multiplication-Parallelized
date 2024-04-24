%%sh
cat > SerialCode.c << EOF
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrixMultiplication(int *a, int *b, int *c, int n, int m, int p){
    int *result = (int*) malloc(n*p*sizeof(int));

    for (int i = 0; i < n * p; i++) {
        result[i] = 0;
    }
    for (int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            for(int x = 0; x < m; x++){
                result[i*p+j] += a[i*m+x] * b[x*p+j];
            }
            c[i*p+j] = result[i*p+j];
        }
    }
}

int main() {
    int n = 1000;
    int m = 2000;
    int p = 1000;

    srand(time(NULL));

    int *A = (int *)malloc(n * m * sizeof(int));
    int *B = (int *)malloc(m * p * sizeof(int));
    int *C = (int *)malloc(n * p * sizeof(int));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A[i*m + j] = rand() % 100;

        }

    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            B[i*p + j] = rand() % 100;

        }

    }

    printf("\n");

    clock_t start = clock();

    matrixMultiplication(A, B, C, n, m, p);

    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC * 1000;

    printf("Time spent: %.2f ms\n", time_spent);

    printf("Resultant Matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%d ", C[i*p + j]);
        }
        printf("\n");
    }

    return 0;
}


EOF
ls -l