#include "matrix_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void usage(const char* prog) {
  fprintf(stderr,
          "Usage:\n"
          "  %s                    → матриці 87×319×213 (варіант 20, scale=1)\n"
          "  %s <scale>            → матриці (87×scale)×(319×scale)×(213×scale)\n"
          "  %s <n1> <n2> <n3>     → довільні розміри A[n1][n2], B[n2][n3], C[n1][n3]\n",
          prog, prog, prog);
}

int main(int argc, char** argv) {
  const int n1_base = 87;
  const int n2_base = 319;
  const int n3_base = 213;

  int n1, n2, n3;
  int write_files = 0;

  if (argc == 1) {
    n1 = n1_base;
    n2 = n2_base;
    n3 = n3_base;
    write_files = 1;
  } else if (argc == 2) {
    int scale = atoi(argv[1]);
    if (scale < 1) {
      fprintf(stderr, "scale має бути >= 1\n");
      usage(argv[0]);
      return 1;
    }
    n1 = n1_base * scale;
    n2 = n2_base * scale;
    n3 = n3_base * scale;
    write_files = (scale == 1);
  } else if (argc == 4) {
    n1 = atoi(argv[1]);
    n2 = atoi(argv[2]);
    n3 = atoi(argv[3]);
    if (n1 < 1 || n2 < 1 || n3 < 1) {
      fprintf(stderr, "n1, n2, n3 мають бути >= 1\n");
      usage(argv[0]);
      return 1;
    }
    write_files = (n1 <= n1_base && n2 <= n2_base && n3 <= n3_base);
  } else {
    usage(argv[0]);
    return 1;
  }

  Matrix A = mat_alloc(n1, n2);
  Matrix B = mat_alloc(n2, n3);
  Matrix C = mat_alloc(n1, n3);

  mat_fill_random(A, 12345u, 100);
  mat_fill_random(B, 67890u, 100);

  double t0 = now_ms();
  mat_mul(C, A, B);
  double t1 = now_ms();

  printf("SEQUENTIAL n1=%d n2=%d n3=%d time_ms=%.3f\n", n1, n2, n3, (t1 - t0));

  if (write_files) {
    mat_write_tsv("out/sequential_A.tsv", "Matrix A", A);
    mat_write_tsv("out/sequential_B.tsv", "Matrix B", B);
    mat_write_tsv("out/sequential_C.tsv", "Matrix C", C);
  }

  mat_free(&A);
  mat_free(&B);
  mat_free(&C);
  return 0;
}

